"""Training data for a Speculative-Forcing-style learned verifier on the 14B causal-fast (4-step schedule, ring KV, FP8, Sage).
Target-only rollout; at each chunk and step k = 1..3 save (a) the last transformer block's hidden state, 2x2 spatially pooled and projected with a
fixed Gaussian matrix 5120 -> 1536 (fp16), (b) the oracle gap rel_k(i) = ||x0^(k)(i) - x0^(4)(i)|| / sigma_t per 1x2x2 token, pooled 2x2 (mean) to the
same grid, (c) the same gap unpooled for evaluation. One file per chunk: <out>/<task>_s<seed>_c<c>.pt = {h: [3, F, H/4, W/4, 1536] fp16,
gap: [3, F, H/2, W/2] fp16, gap_pool: [3, F, H/4, W/4] fp16, sigma, task, seed, chunk}. Scenes are the camera-task set (90 scenes); hold out
sand_filling_up_jar / sandcastle_wave_impact / every 10th task for testing (head_train.py).
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.gap_dump --shard i/n --out DIR [--seed 7]"""
import argparse, os, sys, json, time, numpy as np, torch, torch.nn.functional as F
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--shard", default="0/1"); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--tasks", default="", help="comma list; default = all camera tasks"); ap.add_argument("--proj-dim", type=int, default=1536); ap.add_argument("--norm", default="", help="path of a chan_norm.pt (per-channel mean/std of the raw hidden); computed from the first chunk and saved if missing"); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
tasks = a.tasks.split(",") if a.tasks else sorted(ALL); si, sn = (int(x) for x in a.shard.split("/")); tasks = tasks[si::sn]
B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(B.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 64
for blk in B.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
os.makedirs(a.out, exist_ok=True); dev = "cuda:0"
PROJ = torch.randn(5120, a.proj_dim, generator=torch.Generator().manual_seed(0)).to(dev, torch.bfloat16) / a.proj_dim ** 0.5; torch.save(PROJ.cpu(), a.out + "/proj.pt")
H = {}; NORM = {}
def hook(m, inp, out): H["h"] = out[0] if isinstance(out, tuple) else out
def feat(h):   # raw [L, 5120] -> per-channel standardized (massive-activation channels would otherwise swamp the random projection in fp16) -> projected
    if "mu" not in NORM:
        if a.norm and os.path.exists(a.norm): n = torch.load(a.norm); NORM["mu"], NORM["sd"] = n["mu"].to(dev), n["sd"].to(dev)
        else: NORM["mu"] = h.float().mean(0); NORM["sd"] = h.float().std(0).clamp_min(1e-3); torch.save(dict(mu=NORM["mu"].cpu(), sd=NORM["sd"].cpu()), a.out + "/chan_norm.pt")
    return (((h.float() - NORM["mu"]) / NORM["sd"]).to(torch.bfloat16) @ PROJ)
B.target_model.blocks[-1].register_forward_hook(hook)
def patch(x):
    c, f, h, w = x.shape; return x.view(c, f, 1, h // 2, 2, w // 2, 2).permute(1, 3, 5, 2, 4, 6, 0).reshape(f * (h // 2) * (w // 2), 64)
def gen(seed, c): return torch.Generator().manual_seed(seed * 1009 + c * 7)
done = 0
for task in tasks:
    T = ALL[task]; seed = a.seed
    if os.path.exists("%s/%s_s%d_c12.pt" % (a.out, task, seed)): continue
    t0 = time.time()
    try:
        with torch.no_grad():
            cond = B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); B.reset_caches(); N = cond["num_chunks"]; sig = [float(s) for s in B.node_sigmas()]
            for c in range(N):
                g = gen(seed, c); z = B.chunk_noise(c, g); xs = []; hs = []
                for i in range(len(sig)):
                    x = B.target_x0(z, i, c).float(); xs.append(x)
                    if i < len(sig) - 1:
                        hs.append(feat(H["h"][0]))   # [L, proj] at step i+1 (k = i + 1)
                        s_ = sig[i + 1]; z = (1 - s_) * x + s_ * B.randn_like(x, g)
                f_, h2, w2 = xs[-1].shape[1], xs[-1].shape[2] // 2, xs[-1].shape[3] // 2
                xK = patch(xs[-1]); sg = float(xK.norm(dim=1).median())
                gaps = torch.stack([((patch(xs[k]) - xK).norm(dim=1) / sg).view(f_, h2, w2) for k in range(len(sig) - 1)])          # [3, F, H/2, W/2]
                hp = torch.stack([F.avg_pool2d(h.view(f_, h2, w2, -1).permute(0, 3, 1, 2).float(), 2, ceil_mode=True).permute(0, 2, 3, 1) for h in hs])   # [3, F, H/4, W/4, proj]
                gp = F.avg_pool2d(gaps.view(-1, 1, h2, w2), 2, ceil_mode=True).view(gaps.shape[0], f_, -1, hp.shape[3])
                torch.save(dict(h=hp.half().cpu(), gap=gaps.half().cpu(), gap_pool=gp.half().cpu(), sigma=sg, task=task, seed=seed, chunk=c), "%s/%s_s%d_c%02d.pt" % (a.out, task, seed, c))
                B.commit(xs[-1], c)
        done += 1; print("[GAPDUMP] %s %.0fs (%d)" % (task, time.time() - t0, done), flush=True)
    except Exception as e: print("[GAPDUMP_FAIL] %s %r" % (task, e), flush=True)
print("[GAPDUMP_DONE] shard %s" % a.shard, flush=True)
