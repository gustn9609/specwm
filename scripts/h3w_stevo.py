#!/usr/bin/env python3
"""MiniMax-H3 (plain FL2VA) / H3-World (LoRA, per-latent keyboard actions) generation via DiffSynth (H3-World 패치 파이프라인), 1 GPU.

--mode h3       : LoRA 없음, prompt = video_WM (occlusion 지시 포함), 첫 프레임 조건.  (StEvo "video model" 프로토콜)
--mode h3world  : LoRA + action_script. --tasks 의 pose 문자열(s-3, right-30, left-30)을 H3 latent 격자에 비례 매핑해 keys9 로 변환
                  (s -> S, right -> L+F, left -> J+F, w -> W).  --pose-npy 를 주면 rt2 연속 포즈를 Δyaw/Δpitch/전진 임계값으로 이산화.
출력: <out>/<id>.mp4 + <out>/timing.json.<shard> (파이프라인 호출 wall 초 = 디노이즈 + VAE 디코드 포함)
"""
import argparse, json, os, sys, time, math
import numpy as np, torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 512; torch._dynamo.config.recompile_limit = 512   # flex_attention 이 프롬프트 길이마다 재컴파일 → 기본 한도(8) 초과 시 실패
S = "/home/hk28965/specwm"; H3W = S + "/h3world/H3-World"
sys.path.insert(0, H3W + "/code/abot"); sys.path.insert(0, S + "/h3world/DiffSynth-Studio-h3-v2")
import abot_action as A, action_script as SC
from PIL import Image


def keys_from_pose_string(pose, latent_t):
    toks = [t.strip().split("-") for t in pose.split(",") if t.strip()]; tot = sum(int(n) for _, n in toks)
    k9 = np.zeros((latent_t, len(SC.KEYS9)), np.int64); pos = 0
    for a, n in toks:
        lo, hi = round(pos / tot * latent_t), round((pos + int(n)) / tot * latent_t); pos += int(n)
        keys = {"s": ["S"], "w": ["W"], "right": ["L", "F"], "left": ["J", "F"], "up": ["K"], "down": ["I"], "a": ["A"], "d": ["D"]}.get(a, [])
        for key in keys: k9[lo:hi, SC.KEYS9.index(key)] = 1
    return k9


def keys_from_poses(P, latent_t, src_fps=16.0, dst_fps=24.0):
    """rt2 poses.npy [N,4,4] (16fps) -> keys9 per H3 latent (24fps 격자). 회전: 임계값 rad/frame(24fps) 0.025(pan)/0.225(sharp), 0.05(pitch); 전진: 정규화 변위 0.075/frame."""
    n = len(P); f = P[:, :3, 2]; t = P[:, :3, 3]
    yaw = np.unwrap(np.arctan2(f[:, 0], f[:, 2])); pitch = np.arcsin(np.clip(-f[:, 1], -1, 1))
    fwd = np.array([np.dot(t[i + 1] - t[i], f[i]) for i in range(n - 1)] + [0.0]); sp = np.median(np.abs(fwd[np.abs(fwd) > 1e-6])) if (np.abs(fwd) > 1e-6).any() else 1.0
    k9 = np.zeros((latent_t, len(SC.KEYS9)), np.int64)
    for k, (a, b) in enumerate(A.frame_spans(latent_t)):
        i0, i1 = min(int(a / dst_fps * src_fps), n - 1), min(max(int(b / dst_fps * src_fps), int(a / dst_fps * src_fps) + 1), n - 1)
        nf = max(b - a, 1); dy = (yaw[i1] - yaw[i0]) / nf; dp = (pitch[i1] - pitch[i0]) / nf; df = fwd[i0:i1].sum() / sp / nf
        if abs(dy) >= 0.025: k9[k, SC.KEYS9.index("J" if dy > 0 else "L")] = 1; k9[k, SC.KEYS9.index("F")] = int(abs(dy) >= 0.225)
        if abs(dp) >= 0.05: k9[k, SC.KEYS9.index("K" if dp > 0 else "I")] = 1
        if abs(df) >= 0.075: k9[k, SC.KEYS9.index("W" if df > 0 else "S")] = 1
    return k9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["h3", "h3world"], required=True); ap.add_argument("--tasks", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--num-frames", type=int, default=243); ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--height", type=int, default=480); ap.add_argument("--width", type=int, default=832); ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cfg-scale", type=float, default=None, help="h3world 기본 1.0, h3 는 파이프라인 기본")
    ap.add_argument("--pose-npy", default=None, help="h3world: rt2 poses.npy 로 keys 생성 (tasks 의 pose 문자열 대신)")
    ap.add_argument("--prompt-field", default=None, help="tasks 의 어느 프롬프트를 쓸지 (기본: h3=video_WM, h3world=camera_WM)")
    ap.add_argument("--onload", default="cpu", help="DiffSynth onload_device (cpu=layer 단위 로드, cuda=상주)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); si, sn = [int(x) for x in a.shard.split("/")]
    tasks = [t for i, t in enumerate(json.load(open(a.tasks))) if i % sn == si]
    if (a.num_frames - 5) % 17: raise SystemExit("num_frames must be 17k+5")
    latent_t = A.latent_t_for(a.num_frames)
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
    dtype = torch.bfloat16; dev = "cuda:0"
    vc = dict(offload_dtype=dtype, offload_device="cpu", onload_dtype=dtype, onload_device=a.onload, preparing_dtype=dtype, preparing_device=dev, computation_dtype=dtype, computation_device=dev)
    root = S + "/h3world/DiffSynth-Studio-h3-v2/models"
    cfgs = [ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern=p, **vc) for p in ["FL2VA/text_encoder/model*.safetensors", "FL2VA/transformer/model*.safetensors", "FL2VA/video_vae/source/model.safetensors", "FL2VA/audio_vae/model.safetensors"]]
    t0 = time.time()
    pipe = MiniMaxH3Pipeline.from_pretrained(torch_dtype=dtype, device=dev, model_configs=cfgs, processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"))
    if a.mode == "h3world":
        from safetensors.torch import load_file
        st = load_file(S + "/h3world/checkpoints/H3-World/step-10000.safetensors", device="cpu"); lora = {k: v for k, v in st.items() if ".lora_A." in k or ".lora_B." in k}
        pipe.load_lora(pipe.dit, state_dict=lora, hotload=True); print("[lora] %d pairs" % sum(1 for k in lora if ".lora_A." in k), flush=True)
    print("[load] %.0fs" % (time.time() - t0), flush=True)
    from diffsynth.utils.data.audio_video import write_video_audio
    rep_path = os.path.join(a.out, "timing.json.%d" % si); rep = json.load(open(rep_path)) if os.path.exists(rep_path) else {}

    def load_frame(p):
        img = Image.open(p).convert("RGB")
        if img.size != (a.width, a.height):
            sc = max(a.width / img.width, a.height / img.height); img = img.resize((round(img.width * sc), round(img.height * sc)), Image.LANCZOS)
            l, t_ = (img.width - a.width) // 2, (img.height - a.height) // 2; img = img.crop((l, t_, l + a.width, t_ + a.height))
        return img

    for t in tasks:
        tid = t["id"]; mp4 = os.path.join(a.out, tid + ".mp4")
        if os.path.exists(mp4): print("[skip]", tid, flush=True); continue
        pf = a.prompt_field or ("video_WM" if a.mode == "h3" else "camera_WM"); prompt = t.get(pf, "") or ""
        kw = dict(prompt=prompt, negative_prompt=" ", height=a.height, width=a.width, num_frames=a.num_frames, num_inference_steps=a.steps, seed=a.seed, keyframes=[load_frame(t["image"])], keyframe_indices=[0])
        if a.mode == "h3world":
            k9 = keys_from_poses(np.load(a.pose_npy), latent_t) if a.pose_npy else keys_from_pose_string(t["pose"], latent_t)
            script = SC.annotate_from_keys9(k9); kw.update(action_script=script, negative_action_script=SC.null_script(latent_t), cfg_scale=1.0 if a.cfg_scale is None else a.cfg_scale)
            print("[keys] %s: %s ... %s" % (tid, script[0], script[-1]), flush=True)
        elif a.cfg_scale is not None: kw["cfg_scale"] = a.cfg_scale
        try:
            torch.cuda.synchronize(); tg = time.time(); torch.cuda.reset_peak_memory_stats()
            video, audio = pipe(**kw); torch.cuda.synchronize(); gen_s = time.time() - tg
            tmp = mp4 + ".tmp.mp4"; write_video_audio(video=video, audio=audio, output_path=tmp, fps=24, audio_sample_rate=32000); os.replace(tmp, mp4)
            rep[tid] = dict(mode=a.mode, frames=a.num_frames, hw=[a.height, a.width], steps=a.steps, gen_s=round(gen_s, 1), peak_GiB=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1), prompt=prompt[:120])
            print("[%s] %s" % (tid, rep[tid]), flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); rep[tid] = dict(mode=a.mode, error=str(e)[:300]); torch.cuda.empty_cache()
        json.dump(rep, open(rep_path, "w"), indent=1)
    print("[H3W_DONE] shard %s" % a.shard, flush=True)


if __name__ == "__main__":
    main()
