"""Kernel breakdown of the 1.3B draft forward (denoise and commit) after committing chunks 0..K-1 (fills the KV window like a real rollout).
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.profile_draft_kern --out ../exp/kveval/profile_draft_kern.json"""
import sys, os, json, time, collections, argparse, numpy as np, torch
S = "/home/hk28965/specwm"; ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--K", type=int, default=7); ap.add_argument("--model", default="1.3b"); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding
from torch.profiler import profile, ProfilerActivity
T = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}["sand_filling_up_jar"]; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
D = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), cfg_name="i2v-1.3B", assets_dir=FAST) if a.model == "1.3b" else V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
D.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); D.reset_caches(); sync = torch.cuda.synchronize
with torch.no_grad():
    for ci in range(a.K): x = torch.randn(D._chunk_shape, device="cuda:0"); D.target_x0(x, 0, ci); D.commit(x, ci)
def bucket(n):
    n = n.lower()
    if any(k in n for k in ("flash", "fmha", "attention", "sdpa", "cudnn_sdp", "softmax")): return "attention"
    if any(k in n for k in ("gemm", "cutlass", "sm90", "sm100", "sm120", "matmul", "nvjet", "xmma", "cublas", "addmm", " mm", "linear")): return "gemm"
    if any(k in n for k in ("norm", "rms")): return "norm"
    if any(k in n for k in ("elementwise", "vectorized", "unrolled", "reduce", "copy", "cat", "index", "fill", "pow", "mul", "add", "rotary", "rope", "polar", "complex")): return "elementwise/copy"
    return "other"
R = {}
for name, commit in ((a.model + "_denoise", False), (a.model + "_commit", True)):
    ci = a.K; x = torch.randn(D._chunk_shape, device="cuda:0")
    fn = (lambda: D.commit(x, ci, which="target")) if commit else (lambda: D.target_x0(x, 0, ci))
    with torch.no_grad():
        for _ in range(2): fn()
        sync(); ts = []
        for _ in range(5): t = time.time(); fn(); sync(); ts.append(time.time() - t)
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p: fn(); sync()
    agg = collections.defaultdict(float); top = collections.defaultdict(float)
    for e in p.key_averages():
        ct = getattr(e, "self_device_time_total", 0)
        if ct and e.device_type == torch.autograd.DeviceType.CUDA: agg[bucket(e.key)] += ct / 1e3; top[e.key] += ct / 1e3
    tot = sum(agg.values()); R[name] = dict(wall_ms=round(1000 * float(np.median(ts)), 1), gpu_kernel_ms=round(tot, 1), breakdown_pct={k: round(100 * v / tot, 1) for k, v in sorted(agg.items(), key=lambda kv: -kv[1])}, top=[(k[:100], round(v, 1)) for k, v in sorted(top.items(), key=lambda kv: -kv[1])[:15]])
    print(name, R[name], flush=True)
json.dump(R, open(a.out, "w"), indent=1)
