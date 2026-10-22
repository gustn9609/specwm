"""[10/14, direction 1] Speculation-aware post-training of the LingBot causal-fast models with a two-time FLOW MAP (AnyFlow / MeanFlow recipe,
arXiv 2605.13724): the model learns u(z_t, t, r) so that z_r = z_t - (sigma_t - sigma_r) u for ANY pair t > r, instead of only the instantaneous
velocity. Then every number of steps is a valid sampler, and an early exit is a jump r = 0 at any step (no re-noising).
Stage 1 (this file, 'train'): forward flow-map training on teacher-forced clips (lat.pt of 4-step 14B rollouts + stream actions), LoRA on all
attention/FFN linears + a second timestep embedder delta_embedding = copy of time_embedding, interpolated conditioning
(1-g) emb(t) + g emb'(r), g = 0.25; MeanFlow target u_tgt = v - (sigma_t - sigma_r) du/dt with du/dt by central differences (eps = 5/1000);
(t, r) sampling 50 % r = t (plain flow matching), 25 % r = 0 (consistency), 25 % random; beta08 time weight; adaptive loss rescaling of the
non-diffusion samples to the running diffusion loss. 'eval': flow-map Euler rollouts with N = 1..8 steps and the speculation measure
within-eps(jump to 0 at step k, full N-step path) on held-out clips, videos for the VLM judge.
usage: python -m specwm.flowmap train --model 1p3b --data D1,D2 --val DV --out OUT [--steps 6000 --accum 8 --lr 5e-5]
       python -m specwm.flowmap eval --model 1p3b --lora OUT/lora.pt --val DV --out OUT/eval --N 1,2,3,4,8 [--base 1: no LoRA, r = t Euler]"""
import argparse, os, sys, glob, json, time, math, random, copy, numpy as np, torch, torch.nn as nn
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["train", "eval", "diag"]); ap.add_argument("--model", default="1p3b", choices=["1p3b", "14b"]); ap.add_argument("--data", default=""); ap.add_argument("--val", default=""); ap.add_argument("--out", required=True)
ap.add_argument("--rank", type=int, default=256); ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--accum", type=int, default=8); ap.add_argument("--warmup", type=int, default=500)
ap.add_argument("--gate", type=float, default=0.25); ap.add_argument("--eps-fd", type=float, default=5.0, help="central-difference half step in 1000-timestep units"); ap.add_argument("--p-diff", type=float, default=0.5); ap.add_argument("--p-cons", type=float, default=0.25)
ap.add_argument("--fd-mode", default="fd", choices=["fd", "semi"], help="flow-map target: fd = MeanFlow central-difference d u/dt (eps-fd); semi = discrete semigroup target (delta * v + (t-delta-r) * u_sg(z_{t-delta}, t-delta, r)) / (t-r) along the GT interpolant, no derivative"); ap.add_argument("--delta", type=float, default=0.1, help="semi: step from t towards r in (shifted) sigma units")
ap.add_argument("--delta-sched", type=int, default=0, help="semi: delta = max(delta, (t-r) * (1 - step/S)) — start as plain conditional flow matching for every (t, r), anneal to delta by step S (consistency-training schedule)"); ap.add_argument("--rmode", default="", choices=["", "flow", "native", "t"], help="eval: sampler override (default: native if --base else flow)")
ap.add_argument("--shift", type=float, default=5.0); ap.add_argument("--phase-a", type=int, default=0, help="first N optimizer steps: plain flow matching at random t only (r = t), to smooth the few-step model's velocity field before the flow-map terms"); ap.add_argument("--val-every", type=int, default=500); ap.add_argument("--val-clips", type=int, default=6); ap.add_argument("--eps", type=float, default=0.24); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--lora", default=""); ap.add_argument("--N", default="1,2,3,4,8"); ap.add_argument("--base", type=int, default=0, help="eval: the untrained model with its native sampler (x0 + fresh noise) as the reference row"); ap.add_argument("--video", type=int, default=1); ap.add_argument("--max-chunks", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); random.seed(a.seed); torch.manual_seed(a.seed); dev = torch.device("cuda:0")
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding
from wan.utils.cam_utils import get_Ks_transformed, get_plucker_embeddings
from wan.modules.model import sinusoidal_embedding_1d
from einops import rearrange
FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"; CS = 3
if a.model == "14b": b = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), shift=a.shift)
else: b = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), cfg_name="i2v-1.3B", assets_dir=FAST, shift=a.shift)
_te = b.pipe.text_encoder
class _TE:
    def __init__(s, te): s.te = te; s.model = te.model
    def __call__(s, texts, device): return s.te(texts, dev)
b.pipe.text_encoder = _TE(_te); _te.model.to(dev); m = b.target_model.eval()
for p in m.parameters(): p.requires_grad_(False)
def shift_sig(s): return a.shift * s / (1 + (a.shift - 1) * s)   # the shift-5 schedule: sigma as a function of the uniform time s in [0, 1]
# ---- two-time conditioning: (1-g) emb(t) + g emb'(r); R_T holds the current r (timestep units) and is set before every forward ----
R_T = torch.zeros(1, device=dev); GATE = torch.ones((), device=dev)   # GATE multiplies the LoRA deltas (0 = plain base model)
class TwoTime(nn.Module):
    def __init__(self, te, freq_dim, g): super().__init__(); self.te = te; self.de = copy.deepcopy(te); self.freq_dim = freq_dim; self.g = g
    def forward(self, st):
        sr = sinusoidal_embedding_1d(self.freq_dim, R_T.expand(st.shape[0] * st.shape[1]).float()).view(*st.shape).float()
        return (1 - self.g) * self.te(st) + self.g * self.de(sr)
TT = TwoTime(m.time_embedding, m.freq_dim, a.gate).to(dev); m.time_embedding = TT
for p in TT.de.parameters(): p.requires_grad_(a.cmd == "train")
class LoRA(nn.Module):
    def __init__(self, base, r):
        super().__init__(); self.base = base; self.A = nn.Parameter(torch.randn(r, base.in_features, device=dev) / math.sqrt(base.in_features)); self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
    def forward(self, x): return self.base(x) + (GATE * ((x.float() @ self.A.t()) @ self.B.t())).to(x.dtype)
LOR = []
for blk in m.blocks:
    for nm in ("q", "k", "v", "o"): l = LoRA(getattr(blk.self_attn, nm), a.rank); setattr(blk.self_attn, nm, l); LOR.append(l)
    for i in (0, 2): l = LoRA(blk.ffn[i], a.rank); blk.ffn[i] = l; LOR.append(l)
params = [p for l in LOR for p in (l.A, l.B)] + list(TT.de.parameters())
def load_lora(path):
    ck = torch.load(path, map_location="cpu")
    for l, A_, B_ in zip(LOR, ck["A"], ck["B"]): l.A.data.copy_(A_.to(dev)); l.B.data.copy_(B_.to(dev))
    TT.de.load_state_dict(ck["de"]); print("[fm] loaded %s (step %s)" % (path, ck.get("step")), flush=True)
def save_lora(step, tag): torch.save(dict(rank=a.rank, gate=a.gate, step=step, A=[l.A.detach().cpu() for l in LOR], B=[l.B.detach().cpu() for l in LOR], de={k: v.cpu() for k, v in TT.de.state_dict().items()}), os.path.join(a.out, "lora%s.pt" % tag))
if a.lora: load_lora(a.lora)
if a.base: GATE.fill_(0.0); TT.g = 0.0   # the untrained model: no LoRA, emb(t) only
from torch.utils.checkpoint import checkpoint
for blk in m.blocks:
    _f = blk.forward
    def _ck(*args, _f=_f, **kw): return checkpoint(_f, *args, use_reentrant=False, **kw) if torch.is_grad_enabled() else _f(*args, **kw)
    blk.forward = _ck
TT_ = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
def patch(x):
    c, f, h, w = x.shape; return x.view(c, f, 1, h // 2, 2, w // 2, 2).permute(1, 3, 5, 2, 4, 6, 0).reshape(f * (h // 2) * (w // 2), 64)
def within(xa, xb):
    t = patch(xb); sg = float(t.norm(dim=1).median()); return float(((patch(xa) - t).norm(dim=1) / sg <= a.eps).float().mean())
def emb_chunk(rel, Ks, h, w, lat_h, lat_w):
    f = rel.shape[0]; e = get_plucker_embeddings(rel, Ks, h, w); e = rearrange(e, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)', c1=int(h // lat_h), c2=int(w // lat_w))[None, ...]
    return rearrange(e, 'b (f h w) c -> b c f h w', f=f, h=lat_h, w=lat_w).to(b.param_dtype)
def fwd(z, sig_t, sig_r, ci, grad):
    """u(z, t, r) for the chunk ci (history K/V already committed); sigmas in [0, 1], timestep = sigma * 1000"""
    R_T.fill_(float(sig_r) * 1000.0)
    with torch.set_grad_enabled(grad), torch.amp.autocast("cuda", dtype=b.param_dtype):
        out = m(x=[z.to(b.param_dtype)], t=torch.full((1,), float(sig_t) * 1000.0, device=dev), cross_attn_first_call=not b._cross_init["target"], **b._kwargs("target", ci))[0]
    b._cross_init["target"] = True; return out.float()
@torch.no_grad()
def commit(x0, ci):
    R_T.fill_(0.0); b.commit(x0, ci)
def clip_setup(d):
    meta = json.load(open(d + "/meta.json")); T = TT_[meta["task"]]; stream = meta["stream"]; L = torch.load(d + "/lat.pt").float().to(dev); N = L.shape[1] // CS
    with torch.no_grad(): b.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=meta.get("frames", 161), max_area=meta.get("area", 480 * 832)); b.reset_caches()
    lat_h, lat_w = b._chunk_shape[2], b._chunk_shape[3]; h, w = lat_h * 8, lat_w * 8
    rel = torch.from_numpy(np.load(S + "/data/actstream/%s/rel.npy" % stream)).float().to(dev)[:N * CS]
    Ks = get_Ks_transformed(torch.from_numpy(np.load(S + "/data/actstream/%s/intrinsics.npy" % stream)).float(), height_org=480, width_org=832, height_resize=h, width_resize=w, height_final=h, width_final=w)[0].repeat(N * CS, 1).to(dev)
    b._emb_chunks = [emb_chunk(rel[CS * c:CS * c + CS], Ks[CS * c:CS * c + CS], h, w, lat_h, lat_w) for c in range(N)]
    return meta, L, N
def wbeta(s):   # beta08 weight of AnyFlow on the (shifted) sigma, normalized to mean 1 over the uniform grid
    g = shift_sig(torch.linspace(0, 1, 1001)); y = g * (1 - g).sqrt(); return float((s * (1 - s) ** 0.5) / y.mean())
def gen(seed, c, k=0): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k)
# ---------------- Stage 1 training ----------------
PHASE = [0]; TM = {"setup": 0.0, "loss": 0.0, "commit": 0.0, "opt": 0.0}
def sample_tr():
    u1, u2 = random.random(), random.random(); t, r = max(u1, u2), min(u1, u2); p = random.random()
    if p < a.p_diff or PHASE[0] < a.phase_a: r = t; kind = "diff"
    elif p < a.p_diff + a.p_cons: r = 0.0; kind = "cons"
    else: kind = "map"
    return shift_sig(t), shift_sig(r), kind
def fm_loss(x, ci, g):
    st, sr, kind = sample_tr(); eps = b.randn_like(x, g); v = eps - x; z = (1 - st) * x + st * eps
    u = fwd(z, st, sr, ci, True)
    if kind == "diff": tgt = v
    elif a.fd_mode == "semi":   # [10/14] derivative-free: average velocity over [r, t] = (delta * v + (t' - r) * u(z_t', t', r)) / (t - r) with t' = t - delta on the GT interpolant, u stop-grad
        dl = max(a.delta, (st - sr) * max(0.0, 1.0 - PHASE[0] / a.delta_sched)) if a.delta_sched > 0 else a.delta; dl = min(st - sr, dl); tp = st - dl; zp = z - dl * v
        if tp - sr > 1e-4: up = fwd(zp, tp, sr, ci, False); tgt = (dl * v + (tp - sr) * up) / (st - sr)
        else: tgt = v
        FDMAG.append(float((v - tgt).norm() / v.norm()))
    else:
        e = a.eps_fd / 1000.0; up = fwd(z + v * e, min(st + e, 1.0), sr, ci, False); um = fwd(z - v * e, max(st - e, 0.0), sr, ci, False)
        corr = (st - sr) * (up - um) / (2 * e); tgt = v - corr; FDMAG.append(float(corr.norm() / v.norm()))
    loss = ((u - tgt) ** 2).mean() * wbeta(st); return loss, kind, st, sr
FDMAG = []
@torch.no_grad()
def rollout_chunk(ci, N, seed, exits=(), r_mode="flow"):
    """flow-map Euler over the shift-5 grid with N steps; returns the full-path x and {k: x of the path that jumps to 0 after k forwards}"""
    s = [shift_sig(float(v)) for v in torch.linspace(1.0, 0.0, N + 1)]; z = b.chunk_noise(ci, gen(seed, ci)); outs = {}; gN = gen(seed, ci, 1)   # one generator per chunk: fresh noise at every re-noising step
    for k in range(N):
        st, sr = s[k], s[k + 1]
        if r_mode == "native":   # the stock few-step sampler: x0 estimate, then re-noise with FRESH noise (what the shipped 4-step model was trained for)
            u = fwd(z, st, st, ci, False); x0 = z - st * u
            if k + 1 in exits and k + 1 < N: outs[k + 1] = x0.clone()
            z = (1 - sr) * x0 + sr * b.randn_like(x0, gN) if sr > 0 else x0; continue
        if r_mode == "t": u = fwd(z, st, st, ci, False); x0 = z - st * u
        else:
            u = fwd(z, st, sr, ci, False); x0 = z - st * fwd(z, st, 0.0, ci, False) if (k + 1 in exits and sr > 0) else None
        if k + 1 in exits and k + 1 < N: outs[k + 1] = (x0 if x0 is not None else z - st * u).clone()
        z = z - (st - sr) * u
    outs[N] = z.clone(); return z, outs
@torch.no_grad()
def validate(dirs, N=4, exits=(1, 2, 3), r_mode="flow", tag="val", video=False, max_chunks=0):
    """within-eps(jump-to-0 after k forwards, full N-step path) on teacher-forced held-out chunks + optional free-running videos"""
    agg = {k: [] for k in exits}; losses = []; _ph = PHASE[0]; PHASE[0] = max(_ph, a.phase_a)   # val loss always on the full (t, r) mix, comparable across phases
    for d in dirs:
        meta, L, Nc = clip_setup(d); seed = meta["seed"]; nc = Nc if not max_chunks else min(Nc, max_chunks)
        for c in range(nc):
            if c >= 1:
                xf, outs = rollout_chunk(c, N, seed, exits, r_mode)
                for k in exits: agg[k].append(within(outs[k], xf))
                if a.cmd == "train": l, kind, st, sr = fm_loss(L[:, CS * c:CS * c + CS], c, gen(seed, c, 5)); losses.append(float(l))
            commit(L[:, CS * c:CS * c + CS], c)
    PHASE[0] = _ph; r = {("within_%d" % k): float(np.mean(v)) for k, v in agg.items() if v}; r["loss"] = float(np.mean(losses)) if losses else None; return r
if a.cmd == "train":
    TRAIN = sorted(d for dd in a.data.split(",") for d in glob.glob(dd + "/*_s*") if os.path.exists(d + "/meta.json")); VAL = sorted(d for d in glob.glob(a.val + "/*_s*") if os.path.exists(d + "/meta.json"))[:a.val_clips]
    print("[fm] %s: %d train clips, %d val clips, LoRA r%d %.0fM params + delta embedder %.1fM" % (a.model, len(TRAIN), len(VAL), a.rank, sum(p.numel() for l in LOR for p in (l.A, l.B)) / 1e6, sum(p.numel() for p in TT.de.parameters()) / 1e6), flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0); log = open(os.path.join(a.out, "log.txt"), "a"); step = 0; t0 = time.time(); ema_diff = None; hist = {"diff": [], "cons": [], "map": []}; micro = 0
    def logw(msg): print("[fm]", msg, flush=True); log.write(msg + "\n"); log.flush()
    v = validate(VAL, tag="val0"); logw("step 0 val " + json.dumps(v)); best = v["loss"] if v["loss"] is not None else 1e9
    while step < a.steps:
        TRAIN = sorted(d for dd in a.data.split(",") for d in glob.glob(dd + "/*_s*") if os.path.exists(d + "/meta.json")); random.shuffle(TRAIN); logw("epoch start: %d train clips" % len(TRAIN))
        for d in TRAIN:
            _t = time.time(); meta, L, Nc = clip_setup(d); seed = meta["seed"]; torch.cuda.synchronize(); TM["setup"] += time.time() - _t
            for c in range(Nc):
                x = L[:, CS * c:CS * c + CS]
                if c >= 1:
                    _t = time.time(); loss, kind, st, sr = fm_loss(x, c, gen(seed, c, random.randrange(1 << 20))); lv = float(loss)
                    if kind == "diff": ema_diff = lv if ema_diff is None else 0.98 * ema_diff + 0.02 * lv
                    elif ema_diff is not None: loss = loss * (ema_diff / (lv + 1e-5))   # AnyFlow adaptive reweighting: non-diffusion samples scaled to the diffusion loss magnitude
                    (loss / a.accum).backward(); hist[kind].append(lv); micro += 1; torch.cuda.synchronize(); TM["loss"] += time.time() - _t
                    if micro % a.accum == 0:
                        PHASE[0] = step + 1; _t = time.time()
                        for gp in opt.param_groups: gp["lr"] = a.lr * min(1.0, (step + 1) / a.warmup)
                        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True); step += 1; torch.cuda.synchronize(); TM["opt"] += time.time() - _t
                        if step % 25 == 0: logw("step %d  loss diff %.4f cons %.4f map %.4f  (%d/%d/%d)  |corr|/|v| %.2f  %.0f min  %.1f GiB  t(setup/loss/commit/opt) %.0f/%.0f/%.0f/%.0f s" % (step, np.mean(hist["diff"][-50:] or [0]), np.mean(hist["cons"][-50:] or [0]), np.mean(hist["map"][-50:] or [0]), len(hist["diff"]), len(hist["cons"]), len(hist["map"]), np.mean(FDMAG[-50:] or [0]), (time.time() - t0) / 60, torch.cuda.max_memory_allocated() / 2**30, TM["setup"], TM["loss"], TM["commit"], TM["opt"]))
                        if step % a.val_every == 0:
                            v = validate(VAL); logw("step %d val %s" % (step, json.dumps(v))); save_lora(step, "_s%d" % step)
                            if v["loss"] is not None and v["loss"] < best: best = v["loss"]; save_lora(step, "")
                            clip_setup(d)   # validate() replaced the prepared conditioning: rebuild this clip's and replay its history
                            with torch.no_grad():
                                for cc in range(c + 1): commit(L[:, CS * cc:CS * cc + CS], cc)
                            continue
                        if step >= a.steps: break
                _t = time.time(); commit(x, c); torch.cuda.synchronize(); TM["commit"] += time.time() - _t
            if step >= a.steps: break
    v = validate(VAL); logw("final step %d val %s" % (step, json.dumps(v))); save_lora(step, "_last")
    if v["loss"] is not None and v["loss"] < best: save_lora(step, "")
    print("[FM_DONE] %d steps, %.0f min" % (step, (time.time() - t0) / 60), flush=True)
elif a.cmd == "diag":   # [10/14] is the finite-difference d u/dt a derivative or noise? |corr|/|v| vs eps at fixed (t, r) on teacher-forced val chunks; a true derivative is eps-independent, noise grows as 1/eps
    VAL = sorted(d for d in glob.glob(a.val + "/*_s*") if os.path.exists(d + "/meta.json"))[:a.val_clips]; EPS = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]; rows = []
    with torch.no_grad():
        for d in VAL:
            meta, L, Nc = clip_setup(d); seed = meta["seed"]
            for c in range(min(Nc, a.max_chunks or Nc)):
                x = L[:, CS * c:CS * c + CS]
                if c >= 1:
                    for s_ in (0.3, 0.5, 0.7, 0.9):
                        st = shift_sig(s_); g = gen(seed, c, 11); eps_ = b.randn_like(x, g); v = eps_ - x; z = (1 - st) * x + st * eps_
                        for rr, rn in ((0.0, "r0"), (st / 2, "rhalf")):
                            u0 = fwd(z, st, rr, c, False); ref = None
                            for e1 in EPS:
                                e = e1 / 1000.0; up = fwd(z + v * e, min(st + e, 1.0), rr, c, False); um = fwd(z - v * e, max(st - e, 0.0), rr, c, False); dd = (up - um) / (2 * e)
                                corr = (st - rr) * dd; cos = float((dd * ref).sum() / (dd.norm() * ref.norm() + 1e-9)) if ref is not None else 1.0
                                if ref is None: ref = dd.clone()   # compare every eps to the largest? no: to the first (eps=1); report both ratio and cosine to eps=50 below
                                rows.append(dict(clip=os.path.basename(d), c=c, s=s_, r=rn, eps=e1, ratio=float(corr.norm() / v.norm()), u_minus_v=float((u0 - v).norm() / v.norm()), dd=dd.half().cpu()))
                commit(x, c)
    # cosine of d u/dt between eps and eps=50 (the smoothest estimate) per (clip,c,s,r)
    out = []
    for r_ in rows:
        key = (r_["clip"], r_["c"], r_["s"], r_["r"]); big = [q for q in rows if (q["clip"], q["c"], q["s"], q["r"]) == key and q["eps"] == 50.0][0]["dd"].float()
        dd = r_["dd"].float(); out.append({k: v_ for k, v_ in r_.items() if k != "dd"} | dict(cos50=float((dd * big).sum() / (dd.norm() * big.norm() + 1e-9))))
    json.dump(out, open(a.out + "/diag%s.json" % ("_base" if a.base else ""), "w"), indent=1)
    import collections; agg = collections.defaultdict(list)
    for o in out: agg[(o["s"], o["r"], o["eps"])].append((o["ratio"], o["cos50"], o["u_minus_v"]))
    print("s r eps | |corr|/|v| cos(eps,50) |u-v|/|v|")
    for k in sorted(agg): m = np.mean(agg[k], 0); print("%.1f %s %4.0f | %.2f %.2f %.2f" % (k[0], k[1], k[2], m[0], m[1], m[2]), flush=True)
    print("[FM_DIAG_DONE]", flush=True)
else:   # eval: free-running rollouts with N steps (+ exit paths) on the val clips, videos + within-eps
    VAL = sorted(d for d in glob.glob(a.val + "/*_s*") if os.path.exists(d + "/meta.json"))[:a.val_clips]; Ns = [int(x) for x in a.N.split(",")]; r_mode = a.rmode or ("native" if a.base else "flow"); res = {}
    for d in VAL:
        meta, L, Nc = clip_setup(d); seed = meta["seed"]; name = os.path.basename(d); nc = Nc if not a.max_chunks else min(Nc, a.max_chunks)
        for N in Ns:
            clip_setup(d); lats = []; ex = {k: [] for k in range(1, N)}
            for c in range(nc):
                xf, outs = rollout_chunk(c, N, seed, tuple(range(1, N)), r_mode)
                for k in range(1, N): ex[k].append(within(outs[k], xf))
                commit(xf, c); lats.append(xf.half().cpu())
            od = "%s/%s/N%d%s" % (a.out, name, N, "_base" if a.base else ("_" + a.rmode if a.rmode else "")); os.makedirs(od, exist_ok=True); lat = torch.cat(lats, 1); torch.save(lat, od + "/lat.pt")
            if a.video: b.decode_and_save([lat.float().to(dev)], od)
            res.setdefault(name, {})["N%d" % N] = {("within_%d" % k): float(np.mean(v)) for k, v in ex.items() if v}
            print("[fm-eval] %s N%d %s" % (name, N, json.dumps(res[name]["N%d" % N])), flush=True)
    json.dump(res, open("%s/eval%s.json" % (a.out, "_base" if a.base else ("_" + a.rmode if a.rmode else "")), "w"), indent=1); print("[FM_EVAL_DONE]", flush=True)
