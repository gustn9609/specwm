"""[10/16 frozen baselines] Matrix-Game 2.0 (Skywork, universal distilled ckpt) in ITS setting, timed per PROTOCOL Section 4.
Official stack as in Matrix-Game-2/inference.py: bf16 DiT with flash-attn 2 (flash_attn_func / varlen, FA3 absent on Blackwell), action-module
flex_attention torch.compile'd (max-autotune-no-cudagraphs, their module-level default), fp16 VAE decoder (demo_utils.vae_block3 streaming
decoder with feature cache) torch.compile'd (max-autotune-no-cudagraphs), 352x640, 3 denoising steps [1000, 666, 333] (warped by their shift-5
scheduler), 3 latent frames = 12 pixel frames per chunk, local attention 6 latents (KV window), CLIP-ViT-H image context (no text encoder).
Controls: per-frame keyboard one-hot [fwd, back, left, right] + mouse [pitch, yaw] from a control program (baselines/convert.py: encode_mg2 of
program_to_c2w(P, 12 fps, 129 frames)); their mouse magnitude cap 0.1 (CAM_VALUE) is respected and the saturation is reported in timing.json.
Native playback fps = 12 (utils/visualize.py exports at 12 fps; the paper's "25 FPS" is throughput). 33 latents = 129 frames are generated
(num_frames % 4 == 1 and latents % 3 == 0) and the first 120 frames (10.0 s) are written to video.mp4.
Timing: the official loop is re-implemented verbatim in TimedPipeline.inference_timed with cuda-synchronised stamps around the DiT calls and the
VAE decode of every chunk -> gen_s (DiT only), decode_s, e2e_s (from the loaded PIL image to the last decoded frame, incl. image VAE/CLIP encode),
ttff_s (image -> first decoded chunk = 9 frames), steady fps over chunks 2..n. A warm-up clip (same shapes; compile + autotune) runs first and is
discarded (--warm 1). Peak VRAM allocated / reserved are reset before the measured clip.
usage: python run_mg2.py --image IMG --program P2 --seed 1234 --out DIR [--frames 129] [--keep-frames 120] [--warm 1] [--vae-dtype fp16] [--no-compile]
       python run_mg2.py --jobs JOBS.json [--redo]   (list of {image, program, seed, out}; one model load, warm-up before the first PENDING job;
       [10/17] a job whose out/video.mp4 + timing.json exist is skipped unless --redo, a failing job writes out/ERROR.txt and the batch goes on,
       exit 1 if any job failed; an all-done batch exits 0 before loading the model)"""
import argparse, os, sys, json, time, traceback, numpy as np, torch
S = "/home/hk28965/specwm"; REPO = S + "/tools/mg2/Matrix-Game/Matrix-Game-2"; CKPT = S + "/hf/Matrix-Game-2.0"
sys.path.insert(0, REPO); sys.path.insert(0, S + "/baselines"); os.chdir(REPO)   # their config paths (configs/distilled_model/universal) are relative to the repo
ap = argparse.ArgumentParser(); ap.add_argument("--image"); ap.add_argument("--program"); ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--out"); ap.add_argument("--jobs")
ap.add_argument("--frames", type=int, default=129, help="frames generated: % 4 == 1 and (frames-1)/4 % 3 == 0"); ap.add_argument("--keep-frames", type=int, default=120, help="frames written (10 s at 12 fps)")
ap.add_argument("--fps", type=int, default=12); ap.add_argument("--warm", type=int, default=2); ap.add_argument("--vae-dtype", default="fp16", choices=["fp16", "bf16"]); ap.add_argument("--no-compile", action="store_true")
ap.add_argument("--ckpt", default=CKPT); ap.add_argument("--mouse-scale", type=float, default=None, help="override K (mouse units per degree per frame); default from baselines/calib.json or 0.067")
ap.add_argument("--max-mouse", type=float, default=0.1, help="their CAM_VALUE cap")
ap.add_argument("--redo", action="store_true", help="--jobs: re-run jobs whose out/video.mp4 + timing.json already exist (default: skip them)"); a = ap.parse_args()
# [10/17 batch skip-if-done, additive] decided BEFORE the model load so an all-done batch costs nothing; single-clip mode is unchanged (always runs)
jobs = json.load(open(a.jobs)) if a.jobs else [dict(image=a.image, program=a.program, seed=a.seed, out=a.out)]
def _done(j): o = j["out"].rstrip("/"); return os.path.exists(o + "/video.mp4") and os.path.exists(o + "/timing.json")
pending = [j for j in jobs if not (a.jobs and not a.redo and _done(j))]
if a.jobs:
    print("[MG2] batch %s: %d jobs, %d already done (skipped), %d to run" % (a.jobs, len(jobs), len(jobs) - len(pending), len(pending)), flush=True)
    if not pending: print("[MG2] nothing to do", flush=True); sys.exit(0)
from omegaconf import OmegaConf
from torchvision.transforms import v2
from diffusers.utils import load_image
from einops import rearrange
from safetensors.torch import load_file
import imageio
from pipeline.causal_inference import CausalInferencePipeline, cond_current
from demo_utils.vae_block3 import VAEDecoderWrapper
from demo_utils.constant import ZERO_VAE_CACHE
from wan.vae.wanx_vae import get_wanx_vae_wrapper
from utils.wan_wrapper import WanDiffusionWrapper
from utils.misc import set_seed
import wan.modules.attention as _wattn
from convert import program_to_c2w, encode_mg2, CALIB
def mg2_K():
    try: return float(json.load(open(S + "/baselines/calib.json"))["mg2"]["K_per_deg_per_frame"])
    except Exception: return float(CALIB["mg2"]["K_per_deg_per_frame"])
def encode_program(pid, fps, N, K, cap):
    """shared encoder + saturation report: keyboard (N,4), mouse (N,2) float32, max |mouse| before clipping, fraction of frames clipped"""
    c2w, _ = program_to_c2w(pid, fps, N); cond = encode_mg2(c2w, K, cap); raw = np.abs(np.array(encode_mg2(c2w, K, None)["mouse"], np.float64))
    return dict(keyboard=np.array(cond["keyboard"], np.float32), mouse=np.array(cond["mouse"], np.float32), K=K, max_abs_mouse_unclipped=float(raw.max()), saturated=bool(raw.max() > cap + 1e-9), frac_frames_clipped=float((raw > cap + 1e-9).any(1).mean()))
dev = torch.device("cuda"); dt = torch.bfloat16; vdt = torch.float16 if a.vae_dtype == "fp16" else torch.bfloat16; sync = torch.cuda.synchronize
free0, total0 = torch.cuda.mem_get_info(); others_GiB = (total0 - free0) / 2 ** 30
t_load = time.perf_counter()
config = OmegaConf.load("configs/inference_yaml/inference_universal.yaml"); mode = config.pop("mode"); assert mode == "universal"
generator = WanDiffusionWrapper(**getattr(config, "model_kwargs", {}), is_causal=True)
vdec = VAEDecoderWrapper(); vsd = torch.load(a.ckpt + "/Wan2.1_VAE.pth", map_location="cpu"); vdec.load_state_dict({k: v for k, v in vsd.items() if "decoder." in k or "conv2" in k})
vdec.to(dev, vdt).requires_grad_(False).eval()
if not a.no_compile: vdec.compile(mode="max-autotune-no-cudagraphs")   # official
class TimedPipeline(CausalInferencePipeline):
    @torch.no_grad()
    def inference_timed(self, noise, conditional_dict, mode="universal"):
        """their CausalInferencePipeline.inference, same calls in the same order, plus synchronised stamps per chunk"""
        batch_size, num_channels, num_frames, height, width = noise.shape; assert num_frames % self.num_frame_per_block == 0
        output = torch.zeros([batch_size, num_channels, num_frames, height, width], device=noise.device, dtype=noise.dtype); videos = []
        vae_cache = [None] * len(ZERO_VAE_CACHE); self.kv_cache1 = self.kv_cache_keyboard = self.kv_cache_mouse = self.crossattn_cache = None
        self._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device); self._initialize_kv_cache_mouse_and_keyboard(batch_size=batch_size, dtype=noise.dtype, device=noise.device); self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        cur = 0; dit_s, dec_s, t_first = [], [], None; kw = dict(kv_cache=self.kv_cache1, kv_cache_mouse=self.kv_cache_mouse, kv_cache_keyboard=self.kv_cache_keyboard, crossattn_cache=self.crossattn_cache)
        for _ in range(num_frames // self.num_frame_per_block):
            n = self.num_frame_per_block; noisy_input = noise[:, :, cur:cur + n]; cond = cond_current(conditional_dict, cur, n, mode=mode)
            sync(); t0 = time.perf_counter()
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones([batch_size, n], device=noise.device, dtype=torch.int64) * current_timestep
                _, denoised_pred = self.generator(noisy_image_or_video=noisy_input, conditional_dict=cond, timestep=timestep, current_start=cur * self.frame_seq_length, **kw)
                if index < len(self.denoising_step_list) - 1:
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(rearrange(denoised_pred, 'b c f h w -> (b f) c h w'), torch.randn_like(rearrange(denoised_pred, 'b c f h w -> (b f) c h w')), next_timestep * torch.ones([batch_size * n], device=noise.device, dtype=torch.long))
                    noisy_input = rearrange(noisy_input, '(b f) c h w -> b c f h w', b=denoised_pred.shape[0])
            output[:, :, cur:cur + n] = denoised_pred
            self.generator(noisy_image_or_video=denoised_pred, conditional_dict=cond, timestep=torch.ones_like(timestep) * self.args.context_noise, current_start=cur * self.frame_seq_length, **kw)   # clean-context KV update
            sync(); t1 = time.perf_counter(); cur += n
            video, vae_cache = self.vae_decoder(denoised_pred.transpose(1, 2).to(vdt), *vae_cache)
            sync(); t2 = time.perf_counter(); dit_s.append(t1 - t0); dec_s.append(t2 - t1); videos.append(video)
            if t_first is None: t_first = t2
        return videos, dit_s, dec_s, t_first
pipeline = TimedPipeline(config, generator=generator, vae_decoder=vdec); pipeline.generator.load_state_dict(load_file(a.ckpt + "/base_distilled_model/base_distill.safetensors"))
pipeline = pipeline.to(device=dev, dtype=dt); pipeline.vae_decoder.to(vdt)
vae = get_wanx_vae_wrapper(a.ckpt, torch.float16); vae.requires_grad_(False); vae.eval(); vae = vae.to(dev, dt)
load_s = time.perf_counter() - t_load; print("[MG2] loaded in %.1f s; GPU used by others at start %.1f GiB; flash_attn2=%s fa3=%s" % (load_s, others_GiB, _wattn.FLASH_ATTN_2_AVAILABLE, _wattn.FLASH_ATTN_3_AVAILABLE), flush=True)
H, W = 352, 640; LH, LW = H // 8, W // 8
fp = v2.Compose([v2.Resize(size=(H, W), antialias=True), v2.ToTensor(), v2.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
def resizecrop(image, th, tw):   # their _resizecrop: centre crop to the target aspect, then Resize
    w, h = image.size
    if h / w > th / tw: nw, nh = w, int(w * th / tw)
    else: nh, nw = h, int(h * tw / th)
    return image.crop(((w - nw) / 2, (h - nh) / 2, (w + nw) / 2, (h + nh) / 2))
def run_clip(image_path, pid, seed, out_dir, warm=False):
    N = a.frames; L = (N - 1) // 4 + 1; assert N % 4 == 1 and L % 3 == 0, (N, L)
    enc = encode_program(pid, a.fps, N, a.mouse_scale if a.mouse_scale is not None else mg2_K(), a.max_mouse); kb = torch.tensor(enc["keyboard"]); ms = torch.tensor(enc["mouse"]); assert kb.shape == (N, 4) and ms.shape == (N, 2), (kb.shape, ms.shape)
    pil = load_image(image_path)   # their loader (-> RGB)
    torch.cuda.reset_peak_memory_stats(); set_seed(seed); sync(); t_in = time.perf_counter()
    image = fp(resizecrop(pil, H, W))[None, :, None].to(dtype=dt, device=dev)
    with torch.no_grad():
        pad = torch.zeros_like(image).repeat(1, 1, 4 * (L - 1), 1, 1)
        img_cond = vae.encode(torch.concat([image, pad], dim=2), device=dev, tiled=True, tile_size=[44, 80], tile_stride=[23, 38]).to(dev)
        mask = torch.ones_like(img_cond); mask[:, :, 1:] = 0; cond_concat = torch.cat([mask[:, :4], img_cond], dim=1); visual_context = vae.clip.encode_video(image)
        noise = torch.randn([1, 16, L, LH, LW], device=dev, dtype=dt)
        cond = dict(cond_concat=cond_concat.to(dev, dt), visual_context=visual_context.to(dev, dt), mouse_cond=ms[None].to(dev, dt), keyboard_cond=kb[None].to(dev, dt))
        sync(); t_prep = time.perf_counter()
        videos, dit_s, dec_s, t_first = pipeline.inference_timed(noise, cond, mode=mode); sync(); t_end = time.perf_counter()
        v = rearrange(torch.cat(videos, dim=1), "B T C H W -> B T H W C"); v = ((v.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
    peak_alloc = torch.cuda.max_memory_allocated() / 2 ** 30; peak_res = torch.cuda.max_memory_reserved() / 2 ** 30
    frames_gen = int(v.shape[0]); keep = min(a.keep_frames, frames_gen); chunk_frames = [int(x.shape[1]) for x in videos]
    gen = float(sum(dit_s)); dec = float(sum(dec_s)); steady_gen = sum(chunk_frames[1:]) / sum(dit_s[1:]); steady_e2e = sum(chunk_frames[1:]) / (sum(dit_s[1:]) + sum(dec_s[1:]))
    R = dict(model="Matrix-Game-2.0 universal (base_distilled_model)", ckpt="Skywork/Matrix-Game-2.0@08c42d1a99e57e79df60a964bb815618cdf460f8 base_distilled_model/base_distill.safetensors", repo_commit="SkyworkAI/Matrix-Game@c7a88bfc039cd97695082484986e95b962390676",
             precision="bf16 DiT, %s VAE decoder%s, fp16 image VAE encoder weights run in bf16, CLIP bf16" % (a.vae_dtype, "" if a.no_compile else " + torch.compile(max-autotune-no-cudagraphs)"), attention="flash_attn 2.8.1 (flash_attn_func / varlen) + compiled flex_attention in the action modules; FA3 unavailable",
             vae="Wan2.1 VAE (their demo_utils.vae_block3 streaming decoder with feature cache)", resolution_out="%dx%d" % (v.shape[2], v.shape[1]), frames=keep, frames_generated=frames_gen, fps_native=a.fps, frames_per_call_or_chunk=12, latents_per_chunk=3, steps=[int(x) for x in config.denoising_step_list], steps_warped=[round(float(x), 2) for x in pipeline.denoising_step_list],
             gen_s=round(gen, 3), decode_s=round(dec, 3), e2e_s=round(t_end - t_in, 3), prep_s=round(t_prep - t_in, 3), ttff_s=round(t_first - t_in, 3), steady_fps_gen=round(steady_gen, 2), steady_fps_e2e=round(steady_e2e, 2), fps_gen_all=round(frames_gen / gen, 2), fps_e2e_all=round(frames_gen / (t_end - t_in), 2),
             peak_alloc_GiB=round(peak_alloc, 2), peak_reserved_GiB=round(peak_res, 2), text_encoder_resident="none (no text input); CLIP ViT-H/14 image encoder + xlm-roberta resident on GPU, used once per clip", kv_window="6 latent frames (local_attn_size) = 24 pixel frames", warm=not warm,
             seed=seed, program=pid, image=image_path, control=dict(K_mouse_per_deg_frame=enc["K"], max_abs_mouse_unclipped=round(enc["max_abs_mouse_unclipped"], 4), mouse_cap=a.max_mouse, saturated=bool(enc["saturated"]), frac_frames_clipped=round(enc["frac_frames_clipped"], 3), keys_fwd_back_left_right=enc["keyboard"].sum(0).astype(int).tolist()),
             chunk_dit_s=[round(x, 4) for x in dit_s], chunk_decode_s=[round(x, 4) for x in dec_s], chunk_frames=chunk_frames, load_s=round(load_s, 1), gpu=torch.cuda.get_device_name(0), gpu_mem_used_by_others_at_start_GiB=round(others_GiB, 1), torch=torch.__version__)
    os.makedirs(out_dir, exist_ok=True); imageio.mimwrite(out_dir + "/video.mp4", list(v[:keep]), fps=a.fps, quality=8); json.dump(R, open(out_dir + "/timing.json", "w"), indent=1)
    print("[MG2]%s %s %s seed %d: gen %.2f s (%.1f fps steady), e2e %.2f s (%.1f fps steady), ttff %.2f s, peak %.1f/%.1f GiB, mouse max %.4f%s" % (" WARM" if warm else "", os.path.basename(image_path), pid, seed, gen, steady_gen, t_end - t_in, steady_e2e, t_first - t_in, peak_alloc, peak_res, enc["max_abs_mouse_unclipped"], " SATURATED" if enc["saturated"] else ""), flush=True)
j0 = pending[0]
for w in range(a.warm): run_clip(j0["image"], j0["program"], j0.get("seed", a.seed), j0["out"].rstrip("/") + ("/warmup" if w == 0 else "/warmup%d" % (w + 1)), warm=True)   # torch 2.8 automatic dynamic shapes recompile the decoder's first-chunk graph once more on the 2nd clip -> 2 warm-ups
failures = []
for i, j in enumerate(pending):
    try: run_clip(j["image"], j["program"], j.get("seed", a.seed), j["out"])
    except Exception as e:   # [10/17] batch: log, mark out/ERROR.txt, go on; single clip: raise as before
        if not a.jobs: raise
        failures.append((j["out"], repr(e))); msg = "[MG2] JOB FAILED %d/%d %s: %r\n%s" % (i + 1, len(pending), j["out"], e, traceback.format_exc()); print(msg, flush=True)
        os.makedirs(j["out"], exist_ok=True); open(os.path.join(j["out"], "ERROR.txt"), "w").write(msg); torch.cuda.empty_cache()
if a.jobs:
    print("[MG2] BATCH DONE: %d ok, %d skipped, %d failed of %d jobs" % (len(pending) - len(failures), len(jobs) - len(pending), len(failures), len(jobs)), flush=True)
    for o, e in failures: print("[MG2] FAILED %s: %s" % (o, e), flush=True)
    sys.exit(1 if failures else 0)
