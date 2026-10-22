"""[baselines/sf1p3b] loop.json (spec_early_tf.py) -> OUTDIR/timing.json with the PROTOCOL §4 fields, and video.mp4 copied next to it.
usage: python to_timing.py RUNDIR OUTDIR   (RUNDIR = <out>/<task>/<arm>_s<seed> of spec_early_tf; needs loop.json [+ video.mp4])
Field definitions: gen_s = DiT only (denoise + verifier + clean commit, all chunks; the dense reference pass of --tf-calib is excluded via
gen_method_s); e2e_s = prepare (image/T5/VAE encode) + gen + full VAE decode; ttff_s = from the start of the clip (prepare included) to the first
decoded frame (needs --stream-decode 1, else null); steady_fps_gen = 12 pixel frames per chunk / median chunk wall time over chunks >= 3;
steady_fps_e2e = 12 / median over chunks >= 3 of the interval between the ready times of consecutive chunks' last decoded frames (stream decode)."""
import os, sys, json, shutil, numpy as np
run, out = sys.argv[1], sys.argv[2]; L = json.load(open(run + "/loop.json")); os.makedirs(out, exist_ok=True)
cs = L["chunk_start"] + [L.get("t_gen_end", L["t_gen_start"] + L["gen_s"])]; dt = np.diff(cs); st = dt[3:] if len(dt) > 4 else dt
fpc = 12   # pixel frames per 3-latent-frame chunk (chunk 0 yields 9: 1 + 4 + 4)
ttff = None; fps_e2e = None; fr = L.get("frame_ready")
if fr:   # [(frame id, chunk, ready time)]
    ttff = round(L["prepare_s"] + fr[0][2] - L["t_gen_start"], 3); last = {}
    for f, c, t in fr: last[c] = max(last.get(c, 0), t)
    ks = sorted(last); iv = [last[ks[i]] - last[ks[i - 1]] for i in range(1, len(ks))]; iv = iv[3:] if len(iv) > 4 else iv
    if iv: fps_e2e = round(fpc / float(np.median(iv)), 2)
gm = L.get("gen_method_s", L["gen_s"]); frames = L.get("frames")
T = dict(model="LingBot-World 2.0 1.3B causal-fast + Speculative Forcing training-free verifier (sf1p3b_tf, token-wise early exit)",
         ckpt="lingbot-world-v2-1.3b-causal-fast (HF snapshot 7e36a5f919f86cb4255cc9bfc30adb44963fbde1); T5/VAE/assets from lingbot-world-v2-14b-causal-fast", precision="bf16 weights + activations (no FP8), fp32 x0/verifier",
         attention="SageAttention (dense forwards: step 1, clean commit) / SDPA flash (active-token forwards, fast_patch.ACTIVE)", vae="Wan2.1_VAE.pth (full decode after generation; --stream-decode 1 = second VAE instance decoding every committed chunk on a side stream)",
         resolution_out=L.get("resolution_out"), frames=frames, fps_native=16, frames_per_call_or_chunk=fpc, steps="4-step schedule t=[1000, 937.5, 833.3, 625] with token-wise early exit: mean_steps %.3f" % L["mean_steps"],
         mean_steps=round(L["mean_steps"], 4), gen_s=round(gm, 3), gen_s_wall=L["gen_s"], e2e_s=round(L["prepare_s"] + gm + (L["decode_s"] or 0), 3), prepare_s=L["prepare_s"], decode_s=L["decode_s"], ttff_s=ttff,
         steady_fps_gen=round(fpc / float(np.median(st)), 2), steady_fps_e2e=fps_e2e, chunk_s=[round(float(x), 3) for x in dt], peak_alloc_GiB=L.get("peak_alloc_GiB"), peak_reserved_GiB=L.get("peak_reserved_GiB"),
         text_encoder_resident=True, kv_window=18, kv_sink=6, latent_frames_per_chunk=3, warm=bool(L.get("warm", True)), seed=L["seed"], task=L["task"], mode=L["mode"], tau_red=L.get("tau_red"), tau_stab=L.get("tau_stab"),
         tf_norm=L.get("tf_norm"), verifier="training-free (arXiv 2610.22347 §3.3.1): step 1 redundancy vs clean tokens in the KV window + image latent, steps 2-3 stability; 9x9 spatial + forward temporal Gaussian smoothing; 64-dim latent tokens, distances / median token norm of the draft",
         per_chunk_commit_hist=L.get("per_chunk_commit_hist"), times=L.get("times"), stream_decode=L.get("stream_decode"), source=run)
json.dump(T, open(out + "/timing.json", "w"), indent=1)
if os.path.exists(run + "/video.mp4") and os.path.abspath(run + "/video.mp4") != os.path.abspath(out + "/video.mp4"): shutil.copy(run + "/video.mp4", out + "/video.mp4")
print(json.dumps({k: T[k] for k in ("mean_steps", "gen_s", "e2e_s", "ttff_s", "steady_fps_gen", "steady_fps_e2e", "peak_alloc_GiB", "peak_reserved_GiB", "frames")}))
