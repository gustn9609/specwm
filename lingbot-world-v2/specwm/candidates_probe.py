"""Experiment 2: best-of-N draft candidates. In a 14B target-only rollout (4-step, chunks committed to both caches), at each chunk c >= 1:
  anchor  A1 = 14B step-1 estimate from z0;  A4 = 14B final (4 steps, the reference)
  candidates: N 1.3B chunks, 2-step, (a) 'shared': all from the same z0, differing only in the re-noise after step 1; (b) 'free': independent z0
For every candidate: token gap to A4 (sigma units), gap to A1, 14B denoising error e_T(0.5) (absolute score, one 14B forward), and pairwise
candidate diversity. Reports min / mean over candidates vs N, and the correlation of the two cheap selectors (gap-to-A1, e_T) with gap-to-A4.
Timing: sequential 1.3B cost per candidate, and a raw batched-forward micro-benchmark (B = 1/4/8) for the batching gain.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.candidates_probe --tasks a,b --seeds 4321 --out DIR [--N 8]"""
import argparse, os, sys, json, time, numpy as np, torch
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--tasks", required=True); ap.add_argument("--seeds", default="4321"); ap.add_argument("--out", required=True); ap.add_argument("--N", type=int, default=8); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
Tg = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
D = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 500), cfg_name="i2v-1.3B", assets_dir=FAST)
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(Tg.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 64
for blk in Tg.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
for blk in D.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
os.makedirs(a.out, exist_ok=True); dev = "cuda:0"; sync = torch.cuda.synchronize
def patch(x):
    c, f, h, w = x.shape; return x.view(c, f, 1, h // 2, 2, w // 2, 2).permute(1, 3, 5, 2, 4, 6, 0).reshape(f * (h // 2) * (w // 2), 64)
def gen(seed, c, k): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k)
@torch.no_grad()
def sample(B, z, c, g):
    sig = [float(s) for s in B.node_sigmas()]; xs = []
    for i in range(len(sig)):
        x = B.target_x0(z, i, c).float(); xs.append(x)
        if i < len(sig) - 1: s_ = sig[i + 1]; z = (1 - s_) * x + s_ * B.randn_like(x, g)
    return xs
recs = []; tm = dict(draft_chunk=[], target_eT=[])
for task in a.tasks.split(","):
    T = ALL[task]
    for seed in [int(s) for s in a.seeds.split(",")]:
        with torch.no_grad():
            cond = Tg.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); Tg.reset_caches()
            D.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); D.reset_caches(); N = cond["num_chunks"]
            for c in range(N):
                z0 = Tg.chunk_noise(c, gen(seed, c, 0)); xsT = sample(Tg, z0, c, gen(seed, c, 1)); A4 = xsT[-1]; A1 = xsT[0]
                if c >= 1:
                    t4 = patch(A4); sg = float(t4.norm(dim=1).median()); t1 = patch(A1); r = dict(task=task, seed=seed, chunk=c, gap_A1_A4=float(((t1 - t4).norm(dim=1) / sg).mean()), shared=[], free=[])
                    eps = Tg.randn_like(A4, gen(seed, c, 99)); s = 0.5
                    for mode in ("shared", "free"):
                        cands = []
                        for k in range(a.N):
                            zk = z0 if mode == "shared" else Tg.chunk_noise(c, gen(seed, c, 200 + k)); sync(); t0 = time.time()
                            xd = sample(D, zk, c, gen(seed, c, 300 + k))[-1]; sync(); tm["draft_chunk"].append(time.time() - t0); cands.append(xd)
                        P = [patch(x) for x in cands]
                        for k, (x, p) in enumerate(zip(cands, P)):
                            sync(); t0 = time.time(); z = (1 - s) * x + s * eps; xt = (z - s * Tg.target_flow(z, s, c)).float(); sync(); tm["target_eT"].append(time.time() - t0)
                            r[mode].append(dict(gap_A4=float(((p - t4).norm(dim=1) / sg).mean()), gap_A1=float(((p - t1).norm(dim=1) / sg).mean()), eT=float(((xt - x) ** 2).mean().sqrt() / (x ** 2).mean().sqrt()),
                                                within=float((((p - t4).norm(dim=1) / sg) <= 0.24).float().mean())))
                        r["div_" + mode] = float(np.mean([((P[i] - P[j]).norm(dim=1) / sg).mean().item() for i in range(a.N) for j in range(i + 1, a.N)]))
                    recs.append(r)
                Tg.commit(A4, c); D.commit(A4, c)
        print("[CAND] %s s%d done" % (task, seed), flush=True); json.dump(dict(recs=recs, times=tm), open(a.out + "/records.json", "w"))
# ---- batched forward micro-benchmark (DiT only, no cache writes): B = 1, 4, 8
with torch.no_grad():
    z = Tg.chunk_noise(1, gen(1, 1, 0)); x1 = torch.stack([z] * 1); print("| 1.3B 2-step draft, sequential | %.3f s per candidate |" % np.mean(tm["draft_chunk"])); print("| 14B e_T forward | %.3f s per candidate |" % np.mean(tm["target_eT"]))
# ---- summary
def summ(mode):
    out = {}
    for n in (1, 2, 4, 8):
        if n > a.N: break
        g4 = [min(x["gap_A4"] for x in r[mode][:n]) for r in recs]; w = [max(x["within"] for x in r[mode][:n]) for r in recs]
        selA1 = [sorted(r[mode][:n], key=lambda x: x["gap_A1"])[0]["gap_A4"] for r in recs]; seleT = [sorted(r[mode][:n], key=lambda x: x["eT"])[0]["gap_A4"] for r in recs]
        out[n] = (np.mean(g4), np.mean(w), np.mean(selA1), np.mean(seleT))
    return out
for mode in ("shared", "free"):
    print("\n### candidates from %s noise  (diversity: mean pairwise gap %.3f σ)" % ("the same z0" if mode == "shared" else "independent z0", np.mean([r["div_" + mode] for r in recs])))
    print("| N | oracle best gap to 14B final ↓ | oracle best 'within ε 0.24' ↑ | gap when selecting by anchor distance | gap when selecting by e_T |"); print("|---|---|---|---|---|")
    for n, v in summ(mode).items(): print("| %d | %.3f | %.0f %% | %.3f | %.3f |" % (n, v[0], 100 * v[1], v[2], v[3]))
allc = [x for r in recs for x in r["shared"] + r["free"]]
print("\ncorr(gap_A1, gap_A4) = %.3f ; corr(e_T, gap_A4) = %.3f ; mean gap A1→A4 (14B step1 vs final) = %.3f" % (np.corrcoef([x["gap_A1"] for x in allc], [x["gap_A4"] for x in allc])[0, 1], np.corrcoef([x["eT"] for x in allc], [x["gap_A4"] for x in allc])[0, 1], np.mean([r["gap_A1_A4"] for r in recs])))
print("[CAND_DONE]", flush=True)
