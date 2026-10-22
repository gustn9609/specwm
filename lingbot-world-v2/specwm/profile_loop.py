"""Component profiler for the 1.3B-draft / 14B-target AR loop (cuda-synchronised wall times, one GPU).
Measures: model load, prepare (T5 on CPU, VAE encode, plucker) per model, then per arm (target-only, draft-only, h = target node-0 + draft nodes 1..3,
agree:0:tau) every op of every chunk: noise (CPU randn + H2D), draft forward per node, target forward per node, agreement d, target commit, draft commit,
latent D2H copy; then VAE decode + mp4 write. Finally torch.profiler on one 14B and one 1.3B forward (denoise and commit) grouped into
attention / GEMM / norm+elementwise / other by CUDA kernel time, plus the SDPA backend actually used.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.profile_loop --out ../exp/kveval/profile_loop.json [--task sand_filling_up_jar --seed 1234]"""
import argparse, os, sys, json, time, collections, resource, numpy as np, torch, imageio
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--task", default="sand_filling_up_jar"); ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--out", required=True); ap.add_argument("--frames", type=int, default=161); ap.add_argument("--max-area", type=int, default=480 * 832)
ap.add_argument("--tau", type=float, default=0.21); ap.add_argument("--video-dir", default=""); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; T = ALL[a.task]
FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
sync = torch.cuda.synchronize
def rss_gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
R = dict(task=a.task, seed=a.seed, gpu=torch.cuda.get_device_name(0), load={}, prepare={}, arms={}, kernels={})
t0 = time.time(); D = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), cfg_name="i2v-1.3B", assets_dir=FAST); sync()
R["load"]["draft_1.3B_s"] = round(time.time() - t0, 1); R["load"]["peak_rss_after_draft_GB"] = round(rss_gb(), 1)
t0 = time.time(); Tg = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750)); sync()
R["load"]["target_14B_s"] = round(time.time() - t0, 1); R["load"]["peak_rss_GB"] = round(rss_gb(), 1); R["load"]["gpu_alloc_GB"] = round(torch.cuda.memory_allocated() / 2 ** 30, 1)
print("[load]", R["load"], flush=True)

def timed_prepare(B, name):
    """prepare() split: monkey-patch the text encoder and VAE encode to time them separately."""
    p = B.pipe; te0, enc0 = p.text_encoder, p.vae.encode; tt = {}
    def te(*x, **k): sync(); t = time.time(); r = te0(*x, **k); sync(); tt["t5_cpu_s"] = round(time.time() - t, 2); return r
    def enc(*x, **k): sync(); t = time.time(); r = enc0(*x, **k); sync(); tt["vae_encode_s"] = round(time.time() - t, 2); return r
    p.text_encoder, p.vae.encode = te, enc; sync(); t = time.time()
    cond = B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=a.frames, max_area=a.max_area); sync()
    tt["total_s"] = round(time.time() - t, 2); p.text_encoder, p.vae.encode = te0, enc0; R["prepare"][name] = tt; return cond
cond = timed_prepare(D, "draft"); timed_prepare(Tg, "target"); N = cond["num_chunks"]; sig = [float(s) for s in D.node_sigmas()]
print("[prepare]", R["prepare"], "chunks", N, flush=True)

class Tm:
    def __init__(s): s.t = collections.defaultdict(float); s.n = collections.Counter()
    def __call__(s, key, fn):
        sync(); t = time.time(); r = fn(); sync(); s.t[key] += time.time() - t; s.n[key] += 1; return r

def x0_at(B, z, i, ci): return B.target_x0(z, i, ci)

@torch.no_grad()
def run_arm(arm):
    D.reset_caches(); Tg.reset_caches(); tm = Tm(); g = torch.Generator().manual_seed(a.seed); plan = ""; committed = []; uses_T = arm != "draft"; uses_D = arm != "target"
    sync(); t_all = time.time()
    for ci in range(N):
        z = tm("noise", lambda: D.chunk_noise(ci, g))
        if arm == "target" or ci == 0 and arm != "draft":   # chunk 0 = target (scene establishment), like spec_ar_loop --first target
            x = None
            for i in range(len(sig)):
                x = tm("target_fwd", lambda: x0_at(Tg, z, i, ci))
                if i < len(sig) - 1: s_ = sig[i + 1]; z = tm("noise", lambda: (1 - s_) * x + s_ * Tg.randn_like(x, g))
            plan += "T"
        elif arm == "draft":
            x = None
            for i in range(len(sig)):
                x = tm("draft_fwd", lambda: x0_at(D, z, i, ci))
                if i < len(sig) - 1: s_ = sig[i + 1]; z = tm("noise", lambda: (1 - s_) * x + s_ * D.randn_like(x, g))
            plan += "D"
        else:   # h | agree: node 0 by both models from the same noise
            xd = tm("draft_fwd", lambda: x0_at(D, z, 0, ci)) if arm == "agree" else None
            xt = tm("target_fwd", lambda: x0_at(Tg, z, 0, ci))
            acc = True
            if arm == "agree":
                d = tm("agree_d", lambda: float(((xt.float() - xd.float()) ** 2).mean() / (xt.float() ** 2).mean())); acc = d <= a.tau
            x = xd if (arm == "agree" and acc) else xt; Bf = D if acc else Tg; key = "draft_fwd" if acc else "target_fwd"
            for i in range(1, len(sig)):
                s_ = sig[i]; z = tm("noise", lambda: (1 - s_) * x + s_ * Bf.randn_like(x, g)); x = tm(key, lambda: x0_at(Bf, z, i, ci))
            plan += "D" if acc else "T"
        x = x.float()
        if uses_T: tm("commit_target", lambda: Tg.commit(x, ci))
        if uses_D: tm("commit_draft", lambda: D.commit(x, ci))
        committed.append(tm("d2h", lambda: x.cpu()))
    sync(); gen = time.time() - t_all
    lat = torch.cat(committed, dim=1); sync(); t = time.time(); im = D.pipe.vae.decode([lat.to("cuda:0")])[0].float(); sync(); dec = time.time() - t
    t = time.time(); v = ((im.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).cpu().numpy(); fn = os.path.join(a.video_dir or "/tmp", "prof_%s.mp4" % arm)
    imageio.mimwrite(fn, list((v * 255).astype("uint8")), fps=16, quality=8); wr = time.time() - t
    comp = {k: dict(total_s=round(v, 2), calls=tm.n[k], per_call_ms=round(1000 * v / tm.n[k], 1)) for k, v in sorted(tm.t.items(), key=lambda kv: -kv[1])}
    other = gen - sum(tm.t.values())
    R["arms"][arm] = dict(plan=plan, gen_s=round(gen, 2), components=comp, untimed_overhead_s=round(other, 2), vae_decode_s=round(dec, 2), mp4_write_s=round(wr, 2),
                          end_to_end_s=round(gen + dec + wr, 2), frames=int(im.shape[1]), lat=os.path.basename(fn))
    torch.save(lat.half(), fn.replace(".mp4", "_lat.pt"))
    print("[arm] %s plan=%s gen %.1fs decode %.1fs write %.1fs | %s" % (arm, plan, gen, dec, wr, ", ".join("%s %.1fs/%d" % (k, v["total_s"], v["calls"]) for k, v in comp.items())), flush=True)

run_arm("target")   # warm-up pass is the first arm itself; rerun target at the end to check warm-up effects
for arm in ("draft", "h", "agree", "target"):
    if arm == "target": R["arms"]["target_cold"] = R["arms"].pop("target")
    run_arm(arm)
# fidelity of each arm vs target-only (same noise)
ref = torch.load(os.path.join(a.video_dir or "/tmp", "prof_target_lat.pt")).float()
for arm in ("draft", "h", "agree"):
    x = torch.load(os.path.join(a.video_dir or "/tmp", "prof_%s_lat.pt" % arm)).float(); R["arms"][arm]["rel_l2_to_target"] = round(float((x - ref).norm() / ref.norm()), 4)

# kernel breakdown of single forwards (cache state = after a full target-only run, chunk N-1 slot)
from torch.profiler import profile, ProfilerActivity
def bucket(name):
    n = name.lower()
    if any(k in n for k in ("flash", "fmha", "attention", "sdpa", "cudnn_sdp", "efficient_attention", "softmax")): return "attention"
    if any(k in n for k in ("gemm", "cutlass", "sm90", "sm100", "sm120", "matmul", "nvjet", "xmma", "cublas", "addmm", " mm", "linear")): return "gemm"
    if any(k in n for k in ("norm", "layer_norm", "rms")): return "norm"
    if any(k in n for k in ("elementwise", "vectorized", "unrolled", "reduce", "copy", "cat", "index", "fill", "pow", "mul", "add", "rotary", "rope", "polar", "complex")): return "elementwise/copy"
    return "other"
@torch.no_grad()
def prof(B, name, ci, t_node, commit):
    x = torch.randn(B._chunk_shape, device="cuda:0")
    fn = (lambda: B.commit(x, ci, which="target")) if commit else (lambda: B.target_x0(x, t_node, ci))
    for _ in range(2): fn()
    sync(); ts = []
    for _ in range(3): t = time.time(); fn(); sync(); ts.append(time.time() - t)
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p: fn(); sync()
    agg = collections.defaultdict(float); top = collections.defaultdict(float)
    for e in p.key_averages():
        ct = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
        if ct and e.device_type == torch.autograd.DeviceType.CUDA: agg[bucket(e.key)] += ct / 1e3; top[e.key] += ct / 1e3
    tot = sum(agg.values()); sdpa = sorted({e.key for e in p.key_averages() if "scaled_dot_product" in e.key})
    R["kernels"][name] = dict(wall_ms=round(1000 * float(np.median(ts)), 1), gpu_kernel_ms=round(tot, 1), breakdown_ms={k: round(v, 1) for k, v in sorted(agg.items(), key=lambda kv: -kv[1])},
                              breakdown_pct={k: round(100 * v / tot, 1) for k, v in sorted(agg.items(), key=lambda kv: -kv[1])}, sdpa_ops=sdpa,
                              top_kernels=[(k[:90], round(v, 1)) for k, v in sorted(top.items(), key=lambda kv: -kv[1])[:8]])
    print("[kern]", name, R["kernels"][name]["wall_ms"], "ms wall,", R["kernels"][name]["breakdown_pct"], sdpa, flush=True)
prof(Tg, "target_14B_denoise", N - 1, 0, False); prof(Tg, "target_14B_commit", N - 1, 0, True)
prof(D, "draft_1.3B_denoise", N - 1, 0, False); prof(D, "draft_1.3B_commit", N - 1, 0, True)
R["kernels"]["tokens_per_chunk"] = int(D._seq_len); R["kernels"]["kv_window_tokens"] = int(D._kv_size)
json.dump(R, open(a.out, "w"), indent=1); print("[PROFILE_DONE]", a.out, flush=True)
