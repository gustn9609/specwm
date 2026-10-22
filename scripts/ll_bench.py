#!/usr/bin/env python3
"""A2: LongLive-2.0-5B / LongLive-1.3B 실측 (1 GPU). TTFF(첫 블록 디노이즈+디코드), 정상상태 블록 시간, VAE 포함 총 시간, peak 메모리.
usage (ll-venv, cwd LongLive): python ll_bench.py --config configs/inference.yaml --ckpt ../LongLive-2.0-5B/model_bf16.pt --frames 128 --out ../bench_5b_bf16.json [--width 832 --height 480] [--fp8]"""
import argparse, json, os, sys, time, torch
sys.path.insert(0, os.getcwd())
from omegaconf import OmegaConf
from pipeline import CausalDiffusionInferencePipeline
from utils.config import normalize_config
from utils.inference_utils import load_generator_checkpoint, place_vae_for_streaming, prepare_single_prompt_inputs, save_video
ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--frames", type=int, default=128)
ap.add_argument("--height", type=int, default=None); ap.add_argument("--width", type=int, default=None); ap.add_argument("--out", required=True); ap.add_argument("--prompt", default="A sweeping cinematic journey along the Great Wall of China, winding through golden autumn hills under a brilliant blue sky.")
ap.add_argument("--repeats", type=int, default=2); ap.add_argument("--save", default=None); ap.add_argument("--fp8", action="store_true"); ap.add_argument("--no-compile", action="store_true"); ap.add_argument("--streaming", action="store_true", help="streaming VAE decode (블록마다 디코드; 실시간 배포 설정)")
a = ap.parse_args()
cfg = OmegaConf.load(a.config)
if a.height and a.width:                       # latent = pixel/16 (Wan2.2 5B VAE stride 16), 1.3B: /8
    vs = 16 if "5B" in str(cfg.model_kwargs.model_name) else 8
    cfg.data.image_or_video_shape[3] = a.height // vs; cfg.data.image_or_video_shape[4] = a.width // vs
cfg.data.image_or_video_shape[1] = a.frames; cfg.num_output_frames = a.frames
if a.no_compile: cfg.torch_compile = False
if a.streaming: cfg.inference.streaming_vae = True; cfg.inference.vae_device = None; cfg.inference.vae_type = "mg_lightvae_v2"   # streaming decode 는 LightVAE(Matrix-Game-3.0) 경로만 지원   # 같은 GPU 에서 블록별 디코드 (config 기본 cuda:2 는 단일 GPU 환경에서 invalid)
cfg = normalize_config(cfg); dev = torch.device("cuda"); torch.set_grad_enabled(False)
t0 = time.time(); pipe = CausalDiffusionInferencePipeline(cfg, device=dev); load_generator_checkpoint(pipe.generator, a.ckpt); pipe = pipe.to(device=dev, dtype=torch.bfloat16); place_vae_for_streaming(pipe, cfg); pipe.generator.model.eval().requires_grad_(False)
if a.fp8:
    from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig, PerRow
    quantize_(pipe.generator.model, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()), filter_fn=lambda m, n: isinstance(m, torch.nn.Linear) and m.in_features >= 1024 and m.out_features >= 1024)
print("[load] %.0fs" % (time.time() - t0), flush=True)
fpb = int(cfg.model_kwargs.num_frame_per_block); import importlib; fa_ok = importlib.util.find_spec("flash_attn") is not None
res = dict(ckpt=a.ckpt, flash_attn=fa_ok, streaming_vae=a.streaming, frames=a.frames, frames_per_block=fpb, latent_hw=list(cfg.data.image_or_video_shape[3:]), fp8=a.fp8, runs=[])
# per-block timing: generator forward 호출을 감싸 블록 경계(current_start 변화)로 시간 분할, VAE 는 마지막에 한 번(streaming_vae 가 꺼진 기본 설정) → 총 시간에서 분리
gen = pipe.generator; orig_fwd = gen.forward; marks = []
def fwd(*args, **kw):
    torch.cuda.synchronize(); marks.append((kw.get("current_start", -1), time.time())); return orig_fwd(*args, **kw)
gen.forward = fwd
for r in range(a.repeats):
    marks.clear(); torch.cuda.reset_peak_memory_stats(); g = torch.Generator(device=dev).manual_seed(9999)
    noise, prompts = prepare_single_prompt_inputs(cfg, a.prompt, dev, generator=g)
    torch.cuda.synchronize(); ts = time.time(); video = pipe.inference(noise=noise, text_prompts=prompts); torch.cuda.synchronize(); total = time.time() - ts
    starts = sorted(set(m[0] for m in marks)); first_t = {}; 
    for s, t in marks: first_t.setdefault(s, t)
    bt = [first_t[starts[i + 1]] - first_t[starts[i]] for i in range(len(starts) - 1)]
    gen_end = max(m[1] for m in marks); dec_s = ts + total - gen_end     # 마지막 forward 이후 = 최종 블록 마무리 + VAE 디코드
    nf = int(video.shape[1]); fps_out = 24 if "5B" in str(cfg.model_kwargs.model_name) else 16
    run = dict(total_s=round(total, 2), first_block_s=round(bt[0], 3) if bt else None, ttff_est_s=round(first_t[starts[0]] - ts + (bt[0] if bt else 0) + dec_s * (fpb * 4 + 1) / max(nf, 1), 2), steady_block_s=round(sorted(bt[2:])[len(bt[2:]) // 2], 3) if len(bt) > 3 else None,
               tail_and_decode_s=round(dec_s, 2), px_frames=nf, gen_fps=round(nf / (gen_end - ts), 2), e2e_fps=round(nf / total, 2), peak_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 1), out_fps=fps_out, blocks=len(starts))
    res["runs"].append(run); print("[run %d] %s" % (r, run), flush=True)
    if a.save and r == a.repeats - 1: os.makedirs(os.path.dirname(a.save), exist_ok=True); save_video(video[0], a.save, fps=fps_out)
json.dump(res, open(a.out, "w"), indent=1); print("[LL_BENCH_DONE]", a.out)
