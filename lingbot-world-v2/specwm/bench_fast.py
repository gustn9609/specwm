"""Benchmark fast_patch configs on the full target-only (or draft-only) AR rollout: one model load, one clip per config (same noise),
cumulative flags. Reports gen time, per-forward median, and rel. L2 / max abs diff of the latents vs the baseline run (and baseline vs itself).
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.bench_fast --model 14b --out ../exp/kveval/bench_fast_14b.json [--configs base,base,dedup_t,...]"""
import argparse, os, sys, json, time, numpy as np, torch
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="14b"); ap.add_argument("--out", required=True)
ap.add_argument("--task", default="sand_filling_up_jar"); ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--configs", default="base,base,+dedup_t,+no_kv_cat,+rope_cache,+rope_fp32,+attn=cudnn"); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
T = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}[a.task]; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
if a.model == "14b": B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
else: B = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), cfg_name="i2v-1.3B", assets_dir=FAST)
FP.apply()   # after the binding: its sdpa_patch.apply() would otherwise overwrite our attention hooks
cond = B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); N = cond["num_chunks"]; sig = [float(s) for s in B.node_sigmas()]
sync = torch.cuda.synchronize

@torch.no_grad()
def rollout():
    B.reset_caches(); g = torch.Generator().manual_seed(a.seed); fw = []; out = []; sync(); t0 = time.time()
    for ci in range(N):
        z = B.chunk_noise(ci, g)
        for i in range(len(sig)):
            sync(); t = time.time(); x = B.target_x0(z, i, ci); sync(); fw.append(time.time() - t)
            if i < len(sig) - 1: s_ = sig[i + 1]; z = (1 - s_) * x + s_ * B.randn_like(x, g)
        x = x.float(); sync(); t = time.time(); B.commit(x, ci); sync(); fw.append(time.time() - t); out.append(x.cpu())
    sync(); return time.time() - t0, float(np.median(fw[8:])), torch.cat(out, 1)

R = dict(model=a.model, task=a.task, seed=a.seed, runs=[]); ref = None
for cfg in a.configs.split(","):
    if cfg == "base": FP.FLAGS.update(dedup_t=False, no_kv_cat=False, rope_cache=False, rope_fp32=False, attn="flash")
    else:
        k, _, v = cfg.lstrip("+").partition("=")
        FP.FLAGS[k] = v if v else True
    try:
        gen, fwd, lat = rollout()
    except Exception as ex:
        import traceback; traceback.print_exc(); R["runs"].append(dict(cfg=cfg, flags=dict(FP.FLAGS), error=repr(ex))); FP.FLAGS[k] = False if not v else "flash"; continue
    if ref is None: ref = lat
    d = lat - ref; per = [float(d[:, 3 * c:3 * c + 3].norm() / ref[:, 3 * c:3 * c + 3].norm()) for c in range(N)]
    r = dict(cfg=cfg, flags=dict(FP.FLAGS), gen_s=round(gen, 2), fwd_median_ms=round(1000 * fwd, 1), rel_l2=round(float(d.norm() / ref.norm()), 6),
             max_abs=round(float(d.abs().max()), 5), rel_l2_chunk0=round(per[0], 6), rel_l2_last=round(per[-1], 6))
    R["runs"].append(r); print("[bench]", r, flush=True)
    torch.save(lat.half(), "specwm_out/profile/bench_%s_%d.pt" % (a.model, len(R["runs"])))
json.dump(R, open(a.out, "w"), indent=1); print("[BENCH_DONE]", flush=True)
