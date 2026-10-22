"""Speculative AR loop with every speed patch: draft = LingBot v2 1.3B causal-fast, target = 14B causal-fast, same GPU.
Per chunk (agree verifier, SDVG-style chunk 0 = target): draft node 0 and target node 0 from the same noise; d = ||x0_T - x0_D||^2 / ||x0_T||^2;
accept (d <= tau) -> draft finishes its remaining nodes from its own x0; reject -> target finishes its remaining nodes from its x0. Committed chunk is
written into both KV caches. modes: --tau -1 (target-only, no draft forwards), --tau 1e9 (accept all), else agree.
Profiling: --profile writes a chrome trace + per-phase table (GPU kernel time, wall, GPU idle, host syncs) for the first clip after a warm-up clip.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.spec_fast --tasks a --seeds 1234 --tau 0.21 --out ROOT [--profile] [--t-steps 0,500] [--d-steps 0,250,500,750]"""
import argparse, os, sys, json, time, collections, numpy as np, torch, imageio
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--tasks", default="easy_urban_rain"); ap.add_argument("--seeds", default="1234"); ap.add_argument("--out", required=True)
ap.add_argument("--tau", type=float, default=0.21); ap.add_argument("--t-steps", default="0,500"); ap.add_argument("--d-steps", default="0,250,500,750")
ap.add_argument("--t-fp8", type=int, default=1); ap.add_argument("--d-fp8", type=int, default=0); ap.add_argument("--compile", type=int, default=1); ap.add_argument("--sage", type=int, default=1)
ap.add_argument("--profile", action="store_true"); ap.add_argument("--ring", type=int, default=1); ap.add_argument("--fine", type=int, default=1, help="cuda-sync around every phase (per-phase times); 0 = free-running, total time only"); ap.add_argument("--arm", default=""); ap.add_argument("--no-video", action="store_true"); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
TI = tuple(int(x) for x in a.t_steps.split(",")); DI = tuple(int(x) for x in a.d_steps.split(",")); USE_D = a.tau >= 0
D = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=DI, cfg_name="i2v-1.3B", assets_dir=FAST) if USE_D else None
Tg = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=TI)
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage" if a.sage else "flash", ring_kv=bool(a.ring))
if a.t_fp8: FP.fp8_swap(Tg.target_model, True, keep=False)
if USE_D and a.d_fp8: FP.fp8_swap(D.target_model, True, keep=False)
if a.compile:
    import torch._dynamo as _dyn; _dyn.config.recompile_limit = 64; _dyn.config.accumulated_recompile_limit = 4096
    for B in [x for x in (D, Tg) if x is not None]:
        for blk in B.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
for B in [x for x in (D, Tg) if x is not None]:   # T5 on GPU (shared encoder weights are identical; each binding has its own)
    B.pipe.text_encoder.model.to("cuda:0"); _te = B.pipe.text_encoder
    class _TE:
        def __init__(s, te): s.te = te; s.model = te.model
        def __call__(s, texts, device): return s.te(texts, torch.device("cuda:0"))
    B.pipe.text_encoder = _TE(_te)
sync = torch.cuda.synchronize
FINE = bool(a.fine)
from torch.profiler import record_function as RF

class Tm:
    def __init__(s): s.t = collections.defaultdict(float); s.n = collections.Counter()
    def __call__(s, key, fn):
        if not FINE:
            with RF(key): return fn()
        sync(); t = time.time()
        with RF(key): r = fn(); sync()
        s.t[key] += time.time() - t; s.n[key] += 1; return r

@torch.no_grad()
def run_case(task, seed, out):
    T = ALL[task]; tm = Tm(); sync(); t0 = time.time()
    cond = Tg.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); Tg.reset_caches()
    if USE_D: D.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); D.reset_caches()
    sync(); prep = time.time() - t0
    N = cond["num_chunks"]; st = [float(s) for s in Tg.node_sigmas()]; sd = [float(s) for s in D.node_sigmas()] if USE_D else None
    g = torch.Generator().manual_seed(seed); plan = ""; lat = []; recs = []; sync(); t0 = time.time()
    def finish(B, x, sig, ci, key):
        for i in range(1, len(sig)):
            s_ = sig[i]; z = tm("noise", lambda: (1 - s_) * x + s_ * B.randn_like(x, g)); x = tm(key, lambda: B.target_x0(z, i, ci))
        return x
    for ci in range(N):
        z = tm("noise", lambda: Tg.chunk_noise(ci, g))
        if not USE_D or ci == 0:
            x = tm("T_denoise", lambda: Tg.target_x0(z, 0, ci)); x = finish(Tg, x, st, ci, "T_denoise"); plan += "T"
        else:
            xd = tm("D_denoise", lambda: D.target_x0(z, 0, ci)); xt = tm("T_verify", lambda: Tg.target_x0(z, 0, ci))
            d = tm("agree_d", lambda: float(((xt - xd) ** 2).mean() / (xt ** 2).mean().clamp_min(1e-8))); acc = d <= a.tau; recs.append(round(d, 4))
            x = finish(D, xd, sd, ci, "D_denoise") if acc else finish(Tg, xt, st, ci, "T_denoise"); plan += "D" if acc else "T"
        x = x.float(); tm("T_commit", lambda: Tg.commit(x, ci))
        if USE_D: tm("D_commit", lambda: D.commit(x, ci))
        lat.append(tm("d2h", lambda: x.cpu()))
    sync(); gen = time.time() - t0; L = torch.cat(lat, 1); dec = None
    os.makedirs(out, exist_ok=True); torch.save(L.half(), out + "/lat.pt")
    if not a.no_video:
        sync(); t1 = time.time()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16): im = Tg.pipe.vae.decode([L.cuda()])[0].float()
        sync(); dec = time.time() - t1; v = ((im.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).cpu().numpy(); imageio.mimwrite(out + "/video.mp4", list((v * 255).astype("uint8")), fps=16, quality=8)
    comp = {k: dict(total_s=round(v, 3), calls=tm.n[k], per_call_ms=round(1000 * v / tm.n[k], 1)) for k, v in sorted(tm.t.items(), key=lambda kv: -kv[1])}
    R = dict(task=task, seed=seed, tau=a.tau, t_steps=list(TI), d_steps=list(DI), plan=plan, accepted=plan[1:].count("D"), chunks=N, prepare_s=round(prep, 2), gen_s=round(gen, 2),
             decode_s=None if dec is None else round(dec, 2), untimed_s=round(gen - sum(tm.t.values()), 3), components=comp, d=recs)
    json.dump(R, open(out + "/loop.json", "w"), indent=1)
    print("[SPEC_DONE] %s s%d tau %g plan %s gen %.1fs | %s" % (task, seed, a.tau, plan, gen, ", ".join("%s %.1f/%d" % (k, v["total_s"], v["calls"]) for k, v in comp.items())), flush=True)
    return R

arm = a.arm or ("target" if a.tau < 0 else "spec_tau%g" % a.tau)
tasks = a.tasks.split(","); seeds = [int(x) for x in a.seeds.split(",")]
if a.profile:
    from torch.profiler import profile, ProfilerActivity
    run_case(tasks[0], seeds[0], "%s/%s/%s_warm" % (a.out, tasks[0], arm))   # warm-up: compile every KV length
    sync()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as P:
        R = run_case(tasks[0], seeds[0], "%s/%s/%s_prof" % (a.out, tasks[0], arm))
    P.export_chrome_trace("%s/%s_trace.json" % (a.out, arm))
    # per-phase GPU kernel time and host-sync counts from the record_function ranges
    ev = P.events(); ph = collections.defaultdict(lambda: dict(gpu_ms=0.0, syncs=0, memcpy_d2h=0, kernels=0))
    names = {"T_denoise", "T_verify", "T_commit", "D_denoise", "D_commit", "noise", "agree_d", "d2h"}
    ranges = sorted([(e.time_range.start, e.time_range.end, e.name) for e in ev if e.name in names and e.device_type == torch.autograd.DeviceType.CPU])
    import bisect; starts = [r[0] for r in ranges]
    def owner(ts):
        i = bisect.bisect_right(starts, ts) - 1
        return ranges[i][2] if i >= 0 and ranges[i][0] <= ts <= ranges[i][1] else "other"
    for e in ev:
        if e.device_type == torch.autograd.DeviceType.CUDA:
            o = owner(getattr(e, "linked_correlation_id", None) and e.time_range.start or e.time_range.start)
            ph[o]["gpu_ms"] += (e.time_range.end - e.time_range.start) / 1000; ph[o]["kernels"] += 1
        elif e.name in ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaMemcpyAsync", "aten::item", "aten::_local_scalar_dense"):
            o = owner(e.time_range.start); ph[o]["syncs"] += e.name in ("cudaStreamSynchronize", "aten::_local_scalar_dense")
    json.dump(dict(loop=R, phases=ph), open("%s/%s_profile.json" % (a.out, arm), "w"), indent=1); print("[PROFILE_DONE]", flush=True)
else:
    for task in tasks:
        for seed in seeds:
            out = "%s/%s/%s_s%d" % (a.out, task, arm, seed)
            if os.path.exists(out + "/loop.json"): continue
            run_case(task, seed, out)
    print("[ALL_DONE]", flush=True)
