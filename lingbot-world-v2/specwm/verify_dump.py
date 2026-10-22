"""Data + online fits for cheap verifiers of the 2-GPU speculative pipeline (replace the full 14B node-0 verify forward).
Label = node-0 agreement d0 = ||x0_T(z0) - x0_D(z0)||^2 / ||x0_T||^2 (the agree verifier; Spearman 0.95 with local_err) and local_err.
Rollout is on-policy for agree tau: chunk c is committed (to both models) as the draft chunk if d0 <= tau else the target chunk.
Two cheap verifier families, both fitted by ridge regression on node-0 tokens (online sufficient statistics, no hidden states on disk):
  EE-k : 14B early exit. Run only the first k blocks; token features LN(h_k) (+1) -> 14B node-0 output tokens (64 = 1x2x2 patch x 16 ch).
         d0_hat = rel(x0_D, z - s v_hat). Also "lens-k": the real 14B head applied to h_k (no training).
  DR   : draft-only. Features LN(h_D last block) and the draft output tokens -> 14B node-0 output tokens (an EAGLE-like head on the 1.3B).
mode train: accumulate X^T X, X^T Y per verifier -> <out>/stats_<tag>.pt ; mode test: load <W> (solved by verify_fit.py), record d0_hat of every
verifier per chunk. Both modes save per-chunk records (+ 2x2-pooled draft hidden and per-token error maps for offline nonlinear heads).
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.verify_dump --mode train --tasks a,b --seeds 11,22 --out DIR --tag g6"""
import argparse, os, sys, json, time, math, numpy as np, torch, torch.nn.functional as F
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="train"); ap.add_argument("--tasks", required=True); ap.add_argument("--seeds", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--tag", default="x"); ap.add_argument("--layers", default="4,8,12,16,20,30"); ap.add_argument("--tau", type=float, default=0.21)
ap.add_argument("--W", default=""); ap.add_argument("--pool", type=int, default=1, help="save pooled draft hidden + error maps per chunk"); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
Tg = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 500))
D = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), cfg_name="i2v-1.3B", assets_dir=FAST)
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(Tg.target_model, True, keep=False)
LAY = [int(x) for x in a.layers.split(",")]; os.makedirs(a.out, exist_ok=True); dev = "cuda:0"
CAP = dict(on=None); HT = {}; HD = {}
def hook_T(k):
    def f(m, inp, out):
        if CAP["on"] == "T": HT[k] = out[0] if isinstance(out, tuple) else out
    return f
for k in LAY: Tg.target_model.blocks[k - 1].register_forward_hook(hook_T(k))
def hook_head_T(m, args, out):
    if CAP["on"] == "T": HT["e"] = args[1]; HT["out"] = out
Tg.target_model.head.register_forward_hook(hook_head_T)
def hook_D(m, inp, out):
    if CAP["on"] == "D": HD["h"] = out[0] if isinstance(out, tuple) else out
D.target_model.blocks[-1].register_forward_hook(hook_D)
def hook_head_D(m, args, out):
    if CAP["on"] == "D": HD["out"] = out
D.target_model.head.register_forward_hook(hook_head_D)

def patchify(x):   # [16, F, H, W] latent -> [F*H/2*W/2, 64] tokens in the head's (p q r c) order
    c, f, h, w = x.shape; return x.view(c, f, 1, h // 2, 2, w // 2, 2).permute(1, 3, 5, 2, 4, 6, 0).reshape(f * (h // 2) * (w // 2), 64)
def feat(h, extra=None):
    h = F.layer_norm(h[0].float(), (h.shape[-1],)); cols = [h] + ([extra] if extra is not None else []) + [torch.ones(h.shape[0], 1, device=h.device)]
    return torch.cat(cols, 1)
NAMES = ["EE%d" % k for k in LAY] + ["DR"]
STAT = {}; W = torch.load(a.W, map_location=dev) if a.W else None
def acc(name, X, Y):
    X = X.double(); Y = Y.double()
    if name not in STAT: STAT[name] = dict(xx=torch.zeros(X.shape[1], X.shape[1], dtype=torch.float64, device=dev), xy=torch.zeros(X.shape[1], Y.shape[1], dtype=torch.float64, device=dev), n=0)
    STAT[name]["xx"] += X.T @ X; STAT[name]["xy"] += X.T @ Y; STAT[name]["n"] += X.shape[0]
def gen(seed, c, k): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k)
rel = lambda A, B_: float(((A - B_) ** 2).mean() / (B_ ** 2).mean().clamp_min(1e-8))
sync = torch.cuda.synchronize
@torch.no_grad()
def sample(B, z, sig, c, g, x0=None):
    x = x0; xs = []
    for i in range(len(sig)):
        if not (i == 0 and x0 is not None): x = B.target_x0(z, i, c)
        xs.append(x)
        if i < len(sig) - 1: s_ = float(sig[i + 1]); z = (1 - s_) * x + s_ * B.randn_like(x, g)
    return x.float(), xs
recs = []; tasks = a.tasks.split(","); seeds = [int(s) for s in a.seeds.split(",")]
for task in tasks:
    T = ALL[task]
    for seed in seeds:
        t0 = time.time()
        with torch.no_grad():
            cond = Tg.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); Tg.reset_caches()
            D.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); D.reset_caches()
            N = cond["num_chunks"]; st = [float(s) for s in Tg.node_sigmas()]; sd = [float(s) for s in D.node_sigmas()]; s0 = st[0]
            for c in range(N):
                z0 = Tg.chunk_noise(c, gen(seed, c, 0))
                CAP["on"] = "T"; x0T0 = Tg.target_x0(z0, 0, c); CAP["on"] = None
                xT, _ = sample(Tg, z0, st, c, gen(seed, c, 2), x0=x0T0)
                if c == 0: Tg.commit(xT, c); D.commit(xT, c); continue
                CAP["on"] = "D"; x0D0 = D.target_x0(z0, 0, c); CAP["on"] = None
                xD, xsD = sample(D, z0, sd, c, gen(seed, c, 1), x0=x0D0)
                d0 = rel(x0D0, x0T0); le = rel(xD, xT)
                zt = patchify(z0.float())
                yT = HT["out"][0].float(); yD = HD["out"][0].float(); tD = patchify(x0D0.float()); tT = patchify(x0T0.float())
                assert yT.shape == tT.shape, (yT.shape, tT.shape)
                r = dict(task=task, seed=seed, chunk=c, d0=d0, local_err=le, chain=[rel(xsD[i + 1], xsD[i]) for i in range(len(xsD) - 1)],
                         vnorm=float((yD ** 2).mean()), sigma0=s0)
                fe = {"EE%d" % k: feat(HT[k]) for k in LAY}; fe["DR"] = feat(HD["h"], yD)
                if a.mode == "train":
                    for n in NAMES: acc(n, fe[n], yT)
                else:
                    for n in NAMES:
                        vh = fe[n] @ W[n].float(); x0h = zt - s0 * vh; r["hat_" + n] = float(((tD - x0h) ** 2).mean() / (x0h ** 2).mean().clamp_min(1e-8))
                        r["fit_" + n] = float(((vh - yT) ** 2).mean() / (yT ** 2).mean())
                for k in LAY:   # logit lens: the real head on h_k
                    vh = Tg.target_model.head(HT[k], HT["e"])[0].float(); x0h = zt - s0 * vh
                    r["lens%d" % k] = float(((tD - x0h) ** 2).mean() / (x0h ** 2).mean().clamp_min(1e-8))
                if a.pool:
                    f_, h_, w_ = x0T0.shape[1], x0T0.shape[2] // 2, x0T0.shape[3] // 2
                    hp = F.avg_pool2d(HD["h"][0].float().view(f_, h_, w_, -1).permute(0, 3, 1, 2), 2, ceil_mode=True).permute(0, 2, 3, 1).half().cpu()
                    em = ((tD - tT) ** 2).sum(1).view(f_, h_, w_).cpu(); nm = (tT ** 2).sum(1).view(f_, h_, w_).cpu()
                    torch.save(dict(h=hp, err=em, nrm=nm, d0=d0, le=le), "%s/pool_%s_s%d_c%02d.pt" % (a.out, task, seed, c))
                recs.append(r)
                xc = xD if d0 <= a.tau else xT; r["committed"] = "D" if d0 <= a.tau else "T"
                Tg.commit(xc, c); D.commit(xc, c)
        print("[VDUMP] %s %s s%d %.0fs chunks %d acc %d" % (a.tag, task, seed, time.time() - t0, N, sum(r["committed"] == "D" for r in recs if r["task"] == task and r["seed"] == seed)), flush=True)
        json.dump(recs, open("%s/records_%s_%s.json" % (a.out, a.mode, a.tag), "w"))
        if a.mode == "train": torch.save({n: dict(xx=v["xx"].cpu(), xy=v["xy"].cpu(), n=v["n"]) for n, v in STAT.items()}, "%s/stats_%s.pt" % (a.out, a.tag))
print("[VDUMP_DONE]", a.tag, flush=True)
