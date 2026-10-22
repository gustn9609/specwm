"""Training data for a CROSS-MODEL verifier head: 1.3B draft hidden -> per-token gap to the 14B's 4-step output (same noise, same 14B context).
Rollout = 14B target-only (4-step, FP8/Sage/ring KV), chunks committed to BOTH caches. At every chunk c >= 1 the 1.3B draft runs its 4 steps from the
same z0 (its own re-noising); after each draft step k we save the draft's last-block hidden (channel-standardized with chan_norm_1p3b.pt, dim 1536,
2x2 spatially pooled, fp16) and the label rel_k(i) = ||x0_D^(k)(i) - x0_T^(4)(i)|| / sigma_T (per 1x2x2 token and 2x2-pooled). Also the 14B's own
step-1 gap for reference. One file per chunk: <out>/<task>_s<seed>_c<c>.pt. Same held-out split convention as gap_dump.py (head_train.py handles 4 steps).
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.gap_dump_cross --shard i/n --out DIR [--norm chan_norm_1p3b.pt]"""
import argparse, os, sys, json, time, numpy as np, torch, torch.nn.functional as F
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--shard", default="0/1"); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--tasks", default=""); ap.add_argument("--norm", default="", help="chan_norm for the 1.3B hidden; computed from the first chunk and saved if missing"); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
tasks = a.tasks.split(",") if a.tasks else sorted(ALL); si, sn = (int(x) for x in a.shard.split("/")); tasks = tasks[si::sn]
Tg = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
D = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), cfg_name="i2v-1.3B", assets_dir=FAST)
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(Tg.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 64
for blk in Tg.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
os.makedirs(a.out, exist_ok=True); dev = "cuda:0"; H = {}; NORM = {}
def hook(m, inp, out): H["h"] = out[0] if isinstance(out, tuple) else out
D.target_model.blocks[-1].register_forward_hook(hook)
def feat(h):
    if "mu" not in NORM:
        if a.norm and os.path.exists(a.norm): n = torch.load(a.norm); NORM["mu"], NORM["sd"] = n["mu"].to(dev), n["sd"].to(dev)
        else: NORM["mu"] = h.float().mean(0); NORM["sd"] = h.float().std(0).clamp_min(1e-3); torch.save(dict(mu=NORM["mu"].cpu(), sd=NORM["sd"].cpu()), a.out + "/chan_norm.pt")
    return ((h.float() - NORM["mu"]) / NORM["sd"]).half()
def patch(x):
    c, f, h, w = x.shape; return x.view(c, f, 1, h // 2, 2, w // 2, 2).permute(1, 3, 5, 2, 4, 6, 0).reshape(f * (h // 2) * (w // 2), 64)
def gen(seed, c, k=0): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k)
done = 0
for task in tasks:
    T = ALL[task]; seed = a.seed
    if os.path.exists("%s/%s_s%d_c12.pt" % (a.out, task, seed)): continue
    t0 = time.time()
    try:
        with torch.no_grad():
            cond = Tg.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); Tg.reset_caches()
            D.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); D.reset_caches()
            N = cond["num_chunks"]; st = [float(s) for s in Tg.node_sigmas()]; sd = [float(s) for s in D.node_sigmas()]
            for c in range(N):
                g = gen(seed, c); z0 = Tg.chunk_noise(c, g); z = z0; xsT = []
                for i in range(len(st)):
                    x = Tg.target_x0(z, i, c).float(); xsT.append(x)
                    if i < len(st) - 1: s_ = st[i + 1]; z = (1 - s_) * x + s_ * Tg.randn_like(x, g)
                xT = xsT[-1]
                if c >= 1:
                    gd = gen(seed, c, 1); z = z0; hs = []; xsD = []
                    for i in range(len(sd)):
                        x = D.target_x0(z, i, c).float(); xsD.append(x); hs.append(feat(H["h"][0]))
                        if i < len(sd) - 1: s_ = sd[i + 1]; z = (1 - s_) * x + s_ * D.randn_like(x, gd)
                    f_, h2, w2 = xT.shape[1], xT.shape[2] // 2, xT.shape[3] // 2; tT = patch(xT); sg = float(tT.norm(dim=1).median())
                    gaps = torch.stack([((patch(xd) - tT).norm(dim=1) / sg).view(f_, h2, w2) for xd in xsD])                    # [4, F, h2, w2] draft step k vs 14B final
                    gT1 = ((patch(xsT[0]) - tT).norm(dim=1) / sg).view(f_, h2, w2)                                               # 14B's own step-1 gap (reference)
                    hp = torch.stack([F.avg_pool2d(h.view(f_, h2, w2, -1).permute(0, 3, 1, 2).float(), 2, ceil_mode=True).permute(0, 2, 3, 1) for h in hs])
                    gp = F.avg_pool2d(gaps.view(-1, 1, h2, w2), 2, ceil_mode=True).view(gaps.shape[0], f_, -1, hp.shape[3])
                    torch.save(dict(h=hp.half().cpu(), gap=gaps.half().cpu(), gap_pool=gp.half().cpu(), gapT1=gT1.half().cpu(), sigma=sg, task=task, seed=seed, chunk=c), "%s/%s_s%d_c%02d.pt" % (a.out, task, seed, c))
                Tg.commit(xT, c); D.commit(xT, c)
        done += 1; print("[GAPX] %s %.0fs (%d)" % (task, time.time() - t0, done), flush=True)
    except Exception as e: print("[GAPX_FAIL] %s %r" % (task, e), flush=True)
print("[GAPX_DONE] shard %s" % a.shard, flush=True)
