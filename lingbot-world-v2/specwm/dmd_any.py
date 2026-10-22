"""[10/15] Direction-1 stage 2: DMD re-distillation of the LingBot causal-fast into an ANY-STEP (flow-map) student, AnyFlow / Self-Forcing style.
Student  = causal-fast (v2, GPU 0) + LoRA 'gen' + two-time (t, r) embedding (same layout as specwm/flowmap.py, checkpoints load into `flowmap eval`).
Real score = the bidirectional 14B teacher (v2 bid in v1 layout, lingbot-world-v2/bid_as_v1_cam; two experts, CFG guide 5) on GPU 1, forward only.
Fake score = the same causal-fast base + LoRA 'fake' (r = t), trained by flow matching on the student's own rollouts (clean causal context).
Per generator step: free-running rollout with N ~ {1,2,3,4} flow-map Euler steps (no grad) -> DMD gradient on the whole clip at one noise level
(x0_fake - x0_real, normalised) -> replay the LAST denoising step of `grad-chunks` chunks with grad and regress to (x - grad) (Self-Forcing: gradient
through the last step only; each chunk backward before the next commit) -> + the flow-map co-training loss (semigroup target, annealed delta) on
teacher-forced 14B rollouts (--data). Then `critic-ratio` fake-score steps on fresh rollouts.
usage: CUDA_VISIBLE_DEVICES=g0,g1 python -m specwm.dmd_any --out OUT --data D1,D2 --val DV [--steps 500 --frames 81]"""
import argparse, os, sys, glob, json, time, math, random, copy, importlib, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--data", default=""); ap.add_argument("--val", default="")
ap.add_argument("--frames", type=int, default=81); ap.add_argument("--area", type=int, default=480 * 832); ap.add_argument("--steps", type=int, default=500)
ap.add_argument("--critic-ratio", type=int, default=3); ap.add_argument("--lr-gen", type=float, default=1e-5); ap.add_argument("--lr-fake", type=float, default=2e-5)
ap.add_argument("--rank", type=int, default=256); ap.add_argument("--gate", type=float, default=0.25); ap.add_argument("--guide", type=float, default=5.0)
ap.add_argument("--dmd-weight", type=float, default=1.0); ap.add_argument("--fm-weight", type=float, default=1.0); ap.add_argument("--fm-chunks", type=int, default=4)
ap.add_argument("--delta", type=float, default=0.1); ap.add_argument("--delta-sched", type=int, default=1000); ap.add_argument("--p-diff", type=float, default=0.5); ap.add_argument("--p-cons", type=float, default=0.25)
ap.add_argument("--grad-chunks", type=int, default=3); ap.add_argument("--Ns", default="1,2,3,4"); ap.add_argument("--smin", type=float, default=0.02); ap.add_argument("--smax", type=float, default=0.98)
ap.add_argument("--shift", type=float, default=5.0); ap.add_argument("--val-every", type=int, default=100); ap.add_argument("--val-clips", type=int, default=6); ap.add_argument("--eps", type=float, default=0.24)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--log-every", type=int, default=10); ap.add_argument("--init-lora", default=""); ap.add_argument("--resume", default="")
ap.add_argument("--streams", default="tau1_s0,tau1_s1,tau1_s2,tau1_s3,tau2_s0,tau2_s1,tau2_s2,tau2_s3,tau2j03_s0,tau2j03_s1,tau2j1_s0,tau2j1_s1,tau4_s0,tau4_s1,tau4_s2,tau4_s3")
ap.add_argument("--tasks", default=S + "/lingbot-world-v2/specwm_out/fm_data_train_scenes.txt"); ap.add_argument("--smoke", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); random.seed(a.seed); torch.manual_seed(a.seed)
dev = torch.device("cuda:0"); dev1 = torch.device("cuda:1"); CS = 3
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding
from wan.utils.cam_utils import get_Ks_transformed, get_plucker_embeddings
from wan.modules.model import sinusoidal_embedding_1d
from einops import rearrange
FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"; TT_ = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
b = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), cfg_name="i2v-1.3B", assets_dir=FAST, shift=a.shift)
_te = b.pipe.text_encoder
class _TE:
    def __init__(s, te): s.te = te; s.model = te.model
    def __call__(s, texts, device): return s.te(texts, dev)
b.pipe.text_encoder = _TE(_te); _te.model.to(dev); m = b.target_model.eval()
for p in m.parameters(): p.requires_grad_(False)
def shift_sig(s): return a.shift * s / (1 + (a.shift - 1) * s)
# ---- adapters: 'gen' (student, with the two-time embedding) / 'fake' (critic, r = t, plain embedding) ----
ADAPTER = ["gen"]; R_T = torch.zeros(1, device=dev); GATE = torch.ones((), device=dev)
class TwoTime(nn.Module):
    def __init__(self, te, freq_dim, g): super().__init__(); self.te = te; self.de = copy.deepcopy(te); self.freq_dim = freq_dim; self.g = g
    def forward(self, st):
        if ADAPTER[0] == "fake" or self.g == 0: return self.te(st)
        sr = sinusoidal_embedding_1d(self.freq_dim, R_T.expand(st.shape[0] * st.shape[1]).float()).view(*st.shape).float()
        return (1 - self.g) * self.te(st) + self.g * self.de(sr)
TT = TwoTime(m.time_embedding, m.freq_dim, a.gate); m.time_embedding = TT
for p in TT.de.parameters(): p.requires_grad_(True)
class LoRA(nn.Module):
    def __init__(self, base, r):
        super().__init__(); self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=dev) / math.sqrt(base.in_features)); self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
        self.A2 = nn.Parameter(torch.randn(r, base.in_features, device=dev) / math.sqrt(base.in_features)); self.B2 = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
    def forward(self, x):
        A, B = (self.A, self.B) if ADAPTER[0] == "gen" else (self.A2, self.B2)
        return self.base(x) + (GATE * ((x.float() @ A.t()) @ B.t())).to(x.dtype)
LOR = []
for blk in m.blocks:
    for nm in ("q", "k", "v", "o"): l = LoRA(getattr(blk.self_attn, nm), a.rank); setattr(blk.self_attn, nm, l); LOR.append(l)
    for i in (0, 2): l = LoRA(blk.ffn[i], a.rank); blk.ffn[i] = l; LOR.append(l)
P_GEN = [p for l in LOR for p in (l.A, l.B)] + list(TT.de.parameters()); P_FAKE = [p for l in LOR for p in (l.A2, l.B2)]
def save_gen(step, tag): torch.save(dict(rank=a.rank, gate=a.gate, step=step, A=[l.A.detach().cpu() for l in LOR], B=[l.B.detach().cpu() for l in LOR], de={k: v.cpu() for k, v in TT.de.state_dict().items()}), os.path.join(a.out, "lora%s.pt" % tag))
def save_fake(step): torch.save(dict(step=step, A2=[l.A2.detach().cpu() for l in LOR], B2=[l.B2.detach().cpu() for l in LOR]), os.path.join(a.out, "fake_last.pt"))
def load_gen(path):
    ck = torch.load(path, map_location="cpu")
    for l, A_, B_ in zip(LOR, ck["A"], ck["B"]): l.A.data.copy_(A_.to(dev)); l.B.data.copy_(B_.to(dev))
    TT.de.load_state_dict(ck["de"]); print("[dmd] gen loaded %s (step %s)" % (path, ck.get("step")), flush=True); return ck.get("step", 0)
def load_fake(path):
    ck = torch.load(path, map_location="cpu")
    for l, A_, B_ in zip(LOR, ck["A2"], ck["B2"]): l.A2.data.copy_(A_.to(dev)); l.B2.data.copy_(B_.to(dev))
from torch.utils.checkpoint import checkpoint
for blk in m.blocks:
    _f = blk.forward
    def _ck(*args, _f=_f, **kw): return checkpoint(_f, *args, use_reentrant=False, **kw) if torch.is_grad_enabled() else _f(*args, **kw)
    blk.forward = _ck
def patch(x):
    c, f, h, w = x.shape; return x.view(c, f, 1, h // 2, 2, w // 2, 2).permute(1, 3, 5, 2, 4, 6, 0).reshape(f * (h // 2) * (w // 2), 64)
def within(xa, xb):
    t = patch(xb); sg = float(t.norm(dim=1).median()); return float(((patch(xa) - t).norm(dim=1) / sg <= a.eps).float().mean())
def emb_frames(rel, Ks, h, w, lat_h, lat_w):
    f = rel.shape[0]; e = get_plucker_embeddings(rel, Ks, h, w); e = rearrange(e, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)', c1=int(h // lat_h), c2=int(w // lat_w))[None, ...]
    return rearrange(e, 'b (f h w) c -> b c f h w', f=f, h=lat_h, w=lat_w).to(b.param_dtype)
def fwd(z, sig_t, sig_r, ci, grad):
    R_T.fill_(float(sig_r) * 1000.0)
    with torch.set_grad_enabled(grad), torch.amp.autocast("cuda", dtype=b.param_dtype):
        out = m(x=[z.to(b.param_dtype)], t=torch.full((1,), float(sig_t) * 1000.0, device=dev), cross_attn_first_call=not b._cross_init["target"], **b._kwargs("target", ci))[0]
    b._cross_init["target"] = True; return out.float()
@torch.no_grad()
def commit(x0, ci): R_T.fill_(0.0); b.commit(x0, ci)
def stream_cam(stream, N, h, w, lat_h, lat_w):
    rel = torch.from_numpy(np.load(S + "/data/actstream/%s/rel.npy" % stream)).float().to(dev)[:N * CS]
    Ks = get_Ks_transformed(torch.from_numpy(np.load(S + "/data/actstream/%s/intrinsics.npy" % stream)).float(), height_org=480, width_org=832, height_resize=h, width_resize=w, height_final=h, width_final=w)[0].repeat(N * CS, 1).to(dev)
    return rel, Ks
def setup_task(task, stream, frames, area):
    """student conditioning for (task, stream): returns (Nc, lat_h, lat_w, h, w, full-clip plucker [1,C,F,h,w] for the bid)"""
    T = TT_[task]
    with torch.no_grad(): cond = b.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=frames, max_area=area); b.reset_caches()
    N = cond["num_chunks"]; lat_h, lat_w = b._chunk_shape[2], b._chunk_shape[3]; h, w = lat_h * 8, lat_w * 8; rel, Ks = stream_cam(stream, N, h, w, lat_h, lat_w)
    b._emb_chunks = [emb_frames(rel[CS * c:CS * c + CS], Ks[CS * c:CS * c + CS], h, w, lat_h, lat_w) for c in range(N)]
    return N, lat_h, lat_w, h, w, emb_frames(rel, Ks, h, w, lat_h, lat_w)
def clip_setup(d):
    meta = json.load(open(d + "/meta.json")); L = torch.load(d + "/lat.pt").float().to(dev); N = L.shape[1] // CS
    setup_task(meta["task"], meta["stream"], meta.get("frames", 161), meta.get("area", 480 * 832)); return meta, L, N
def gen(seed, c, k=0): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k)
# ---- the bidirectional teacher (v1 layout) on GPU 1: the spec_v0 two-package pattern ----
os.chdir(S + "/lingbot-world"); sys.path = [S + "/lingbot-world"] + [p for p in sys.path if p != S + "/lingbot-world-v2"]
for k in [k for k in sys.modules if k == "wan" or k.startswith("wan.")]: del sys.modules[k]
_spec = importlib.util.spec_from_file_location("specwm_v1", S + "/lingbot-world/specwm/__init__.py", submodule_search_locations=[S + "/lingbot-world/specwm"]); _pkg = importlib.util.module_from_spec(_spec); sys.modules["specwm_v1"] = _pkg; _spec.loader.exec_module(_pkg)
V1Binding = importlib.import_module("specwm_v1.binding_v1").V1Binding
b1 = V1Binding(S + "/lingbot-world-v2/bid_as_v1_cam", t5_cpu=True, device=str(dev1)); b1.device = dev1
b1.pipe.high_noise_model.to(dev1).eval(); b1.pipe.low_noise_model.to(dev1).eval()
TT_ = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
_tasks_all = [t.strip() for t in open(a.tasks) if t.strip()]; _te1 = b1.pipe.text_encoder; _te1.model.to(dev1); b1._txt_cache = {}
with torch.no_grad():   # text embeddings for every training prompt once on the GPU (the fp32 T5 is 23 GB), then T5 back to the CPU
    _ctxn = [t.to(dev1) for t in _te1([b1.pipe.sample_neg_prompt], dev1)]
    for _tk in _tasks_all:
        _pr = TT_[_tk].get("camera_WM", "")
        if (_pr, b1.pipe.sample_neg_prompt) not in b1._txt_cache: b1._txt_cache[(_pr, b1.pipe.sample_neg_prompt)] = ([t.to(dev1) for t in _te1([_pr], dev1)], _ctxn)
_te1.model.cpu(); torch.cuda.empty_cache(); print("[dmd] cached %d prompt embeddings" % len(b1._txt_cache), flush=True)
for mm in (b1.pipe.high_noise_model, b1.pipe.low_noise_model):
    for p in mm.parameters(): p.requires_grad_(False)
os.chdir(S + "/lingbot-world-v2")
print("[dmd] bid 14B on %s: %.1f GiB" % (dev1, torch.cuda.memory_allocated(dev1) / 2**30), flush=True)
BID_COND = {}
def bid_cond(task, stream, plucker, frames, area):
    """[10/15 fix] cache only the per-task part (image latent y, text ctx: ~5 MB); the per-stream plucker (~190 MB fp32 at 81 frames) is moved
    to GPU 1 per call -- caching it per (task, stream) leaked ~0.1 GiB/step and OOMed the first pilot at step ~90"""
    key = (task, frames, area)
    if key not in BID_COND:
        T = TT_[task]
        with torch.no_grad(): c = b1.prepare(T["image"], T.get("camera_WM", ""), None, frame_num=frames, max_area=area)
        BID_COND[key] = c
    c = dict(BID_COND[key]); lat_f = c["lat"][0]; assert plucker.shape[2] == lat_f, (plucker.shape, c["lat"])
    c["dit_cond"] = {"c2ws_plucker_emb": plucker.to(dev1, b1.param_dtype).chunk(1, dim=0)}; return c
@torch.no_grad()
def real_x0(z, sig, c):
    """x0 prediction of the bidirectional teacher (CFG, expert by noise level) for the whole clip z [16,F,h,w] at sigma"""
    t_val = float(sig) * 1000.0; mm = b1.pipe.high_noise_model if t_val >= b1.boundary_t else b1.pipe.low_noise_model
    tt = torch.full((1,), t_val, device=dev1); zin = [z.to(dev1, b1.param_dtype)]
    with torch.amp.autocast("cuda", dtype=b1.param_dtype):
        fc = mm(zin, t=tt, context=[c["ctx"][0]], seq_len=c["seq_len"], y=[c["y"]], dit_cond_dict=c["dit_cond"])[0]
        fu = mm(zin, t=tt, context=c["ctxn"], seq_len=c["seq_len"], y=[c["y"]], dit_cond_dict=c["dit_cond"])[0]
    flow = (fu + a.guide * (fc - fu)).float(); return (z.to(dev1).float() - float(sig) * flow).to(dev)
# ---- student rollout (flow-map Euler, N steps, deterministic) ----
@torch.no_grad()
def rollout(Nc, N, seed):
    """returns x [16, Nc*CS, h, w] (clean chunks, committed as history) and z_in[c] = the latent entering the last step of chunk c"""
    ADAPTER[0] = "gen"; b.reset_caches(); s = [shift_sig(float(v)) for v in torch.linspace(1.0, 0.0, N + 1)]; xs = []; zin = []
    for c in range(Nc):
        z = b.chunk_noise(c, gen(seed, c))
        for k in range(N):
            st, sr = s[k], s[k + 1]
            if k == N - 1: zin.append(z.clone())
            z = z - (st - sr) * fwd(z, st, sr, c, False)
        commit(z, c); xs.append(z)
    return torch.cat(xs, 1), zin, s[N - 1]
def fake_x0_clip(z, x, sig, grad_chunks=(), loss_tgt=None):
    """critic x0 for the noisy clip z, teacher-forced on the clean clip x (causal clean context). With grad_chunks + loss_tgt: flow-matching loss
    on those chunks (backward per chunk) and the per-chunk losses are returned instead."""
    ADAPTER[0] = "fake"; b.reset_caches(); Nc = x.shape[1] // CS; outs = []; losses = []
    for c in range(Nc):
        zc = z[:, CS * c:CS * c + CS]; g = c in grad_chunks
        u = fwd(zc, sig, sig, c, g)
        if g:
            l = Fn.mse_loss(u, loss_tgt[:, CS * c:CS * c + CS]); (l / len(grad_chunks)).backward(); losses.append(float(l))
        else: outs.append((zc - float(sig) * u).detach())
        commit(x[:, CS * c:CS * c + CS], c)
    ADAPTER[0] = "gen"; return (torch.cat(outs, 1) if outs else None), losses
def sample_sig():
    s = random.uniform(0.0, 1.0); return float(min(max(shift_sig(s), a.smin), a.smax))
# ---- flow-map co-training (semigroup target, annealed delta) on teacher-forced 14B rollouts ----
STEP = [0]
def sample_tr():
    u1, u2 = random.random(), random.random(); t, r = max(u1, u2), min(u1, u2); p = random.random()
    if p < a.p_diff: r = t; kind = "diff"
    elif p < a.p_diff + a.p_cons: r = 0.0; kind = "cons"
    else: kind = "map"
    return shift_sig(t), shift_sig(r), kind
def wbeta(s):
    g = shift_sig(torch.linspace(0, 1, 1001)); y = g * (1 - g).sqrt(); return float((s * (1 - s) ** 0.5) / y.mean())
def fm_loss(x, ci, g):
    st, sr, kind = sample_tr(); eps = b.randn_like(x, g); v = eps - x; z = (1 - st) * x + st * eps
    u = fwd(z, st, sr, ci, True)
    if kind == "diff": tgt = v
    else:
        dl = max(a.delta, (st - sr) * max(0.0, 1.0 - STEP[0] / a.delta_sched)) if a.delta_sched > 0 else a.delta; dl = min(st - sr, dl); tp = st - dl; zp = z - dl * v
        if tp - sr > 1e-4: up = fwd(zp, tp, sr, ci, False); tgt = (dl * v + (tp - sr) * up) / (st - sr)
        else: tgt = v
    return ((u - tgt) ** 2).mean() * wbeta(st), kind
def fm_step(d, seed):
    """one teacher-forced clip: loss on `fm-chunks` random chunks (backward per chunk), returns mean loss"""
    ADAPTER[0] = "gen"; meta, L, Nc = clip_setup(d); chosen = set(random.sample(range(1, Nc), min(a.fm_chunks, Nc - 1))); ls = []
    for c in range(Nc):
        x = L[:, CS * c:CS * c + CS]
        if c in chosen:
            l, kind = fm_loss(x, c, gen(seed, c, random.randrange(1 << 20))); (a.fm_weight * l / len(chosen)).backward(); ls.append(float(l))
        commit(x, c)
    return float(np.mean(ls)) if ls else 0.0
@torch.no_grad()
def validate(dirs, N=4, exits=(1, 2, 3)):
    ADAPTER[0] = "gen"; agg = {k: [] for k in exits}
    for d in dirs:
        meta, L, Nc = clip_setup(d); seed = meta["seed"]; s = [shift_sig(float(v)) for v in torch.linspace(1.0, 0.0, N + 1)]
        for c in range(Nc):
            if c >= 1:
                z = b.chunk_noise(c, gen(seed, c)); outs = {}
                for k in range(N):
                    st, sr = s[k], s[k + 1]
                    if k + 1 in exits and k + 1 < N: outs[k + 1] = z - st * fwd(z, st, 0.0, c, False)
                    z = z - (st - sr) * fwd(z, st, sr, c, False)
                for k in exits: agg[k].append(within(outs[k], z))
            commit(L[:, CS * c:CS * c + CS], c)
    return {("within_%d" % k): float(np.mean(v)) for k, v in agg.items() if v}
# ---------------- training ----------------
TASKS = [t.strip() for t in open(a.tasks) if t.strip()]; STREAMS = a.streams.split(","); Ns = [int(x) for x in a.Ns.split(",")]
DATA = sorted(d for dd in a.data.split(",") if dd for d in glob.glob(dd + "/*_s*") if os.path.exists(d + "/meta.json")); VAL = sorted(d for d in glob.glob(a.val + "/*_s*") if os.path.exists(d + "/meta.json"))[:a.val_clips] if a.val else []
opt_g = torch.optim.AdamW(P_GEN, lr=a.lr_gen, betas=(0.9, 0.95), weight_decay=0.0); opt_f = torch.optim.AdamW(P_FAKE, lr=a.lr_fake, betas=(0.9, 0.95), weight_decay=0.0)
step = 0
if a.init_lora: load_gen(a.init_lora)
if a.resume: step = load_gen(a.resume + "/lora_last.pt"); load_fake(a.resume + "/fake_last.pt")
log = open(os.path.join(a.out, "log.txt"), "a")
def logw(msg): print("[dmd]", msg, flush=True); log.write(msg + "\n"); log.flush()
logw("tasks %d streams %d data clips %d val clips %d; gen params %.0fM fake %.0fM" % (len(TASKS), len(STREAMS), len(DATA), len(VAL), sum(p.numel() for p in P_GEN) / 1e6, sum(p.numel() for p in P_FAKE) / 1e6))
if VAL and not a.smoke: logw("step %d val %s" % (step, json.dumps(validate(VAL))))
H = dict(dmd=[], gn=[], fake=[], fm=[], tsel=[]); t0 = time.time(); TM = dict(roll=0.0, bid=0.0, fake=0.0, replay=0.0, fm=0.0, critic=0.0)
def pick(): return random.choice(TASKS), random.choice(STREAMS), random.randrange(1 << 30)
while step < a.steps:
    STEP[0] = step
    # ---- critic steps ----
    for _ in range(a.critic_ratio):
        _t = time.time(); task, stream, seed = pick(); Nc, lat_h, lat_w, h, w, pl = setup_task(task, stream, a.frames, a.area); N = random.choice(Ns)
        x, _, _ = rollout(Nc, N, seed); sig = sample_sig(); eps = torch.randn_like(x); z = (1 - sig) * x + sig * eps
        _, ls = fake_x0_clip(z, x, sig, grad_chunks=tuple(range(Nc)), loss_tgt=eps - x)
        torch.nn.utils.clip_grad_norm_(P_FAKE, 1.0); opt_f.step(); opt_f.zero_grad(set_to_none=True); H["fake"].append(float(np.mean(ls))); TM["critic"] += time.time() - _t
    # ---- generator step ----
    _t = time.time(); task, stream, seed = pick(); Nc, lat_h, lat_w, h, w, pl = setup_task(task, stream, a.frames, a.area); N = random.choice(Ns)
    x, zin, s_last = rollout(Nc, N, seed); torch.cuda.synchronize(); TM["roll"] += time.time() - _t
    _t = time.time(); c1 = bid_cond(task, stream, pl, a.frames, a.area); sig = sample_sig(); eps = torch.randn_like(x); z = (1 - sig) * x + sig * eps
    x0_real = real_x0(z, sig, c1); torch.cuda.synchronize(); TM["bid"] += time.time() - _t
    _t = time.time(); x0_fake, _ = fake_x0_clip(z, x, sig); TM["fake"] += time.time() - _t
    with torch.no_grad():
        nrm = (x - x0_real).abs().mean().clamp_min(1e-6); gvec = torch.nan_to_num((x0_fake - x0_real) / nrm); tgt = x - gvec
    H["gn"].append(float(gvec.norm() / x.norm())); H["tsel"].append(sig)
    # replay the last step of `grad-chunks` chunks with grad, backward per chunk, history re-committed from x
    _t = time.time(); ADAPTER[0] = "gen"; b.reset_caches(); G = set(random.sample(range(Nc), min(a.grad_chunks, Nc))); dl = []
    for c in range(Nc):
        if c in G:
            xc = zin[c] - s_last * fwd(zin[c], s_last, 0.0, c, True); l = 0.5 * Fn.mse_loss(xc, tgt[:, CS * c:CS * c + CS]); (a.dmd_weight * l / len(G)).backward(); dl.append(float(l))
        commit(x[:, CS * c:CS * c + CS], c)
    H["dmd"].append(float(np.mean(dl))); torch.cuda.synchronize(); TM["replay"] += time.time() - _t
    if DATA and a.fm_weight > 0:
        _t = time.time(); H["fm"].append(fm_step(random.choice(DATA), seed)); TM["fm"] += time.time() - _t
    torch.nn.utils.clip_grad_norm_(P_GEN, 1.0); opt_g.step(); opt_g.zero_grad(set_to_none=True); step += 1
    if step % a.log_every == 0:
        logw("step %d  N%d sig %.2f  dmd %.4f |grad|/|x| %.3f  fake %.4f  fm %.4f  %.0f min  %.1f/%.1f GiB  t(roll/bid/fake/replay/fm/critic) %.0f/%.0f/%.0f/%.0f/%.0f/%.0f s" % (
            step, N, sig, np.mean(H["dmd"][-a.log_every:]), np.mean(H["gn"][-a.log_every:]), np.mean(H["fake"][-a.log_every * a.critic_ratio:] or [0]), np.mean(H["fm"][-a.log_every:] or [0]), (time.time() - t0) / 60,
            torch.cuda.max_memory_allocated(dev) / 2**30, torch.cuda.max_memory_allocated(dev1) / 2**30, TM["roll"], TM["bid"], TM["fake"], TM["replay"], TM["fm"], TM["critic"]))
    if step % a.val_every == 0 or step >= a.steps:
        save_gen(step, "_s%d" % step); save_gen(step, "_last"); save_fake(step)
        if VAL: logw("step %d val %s" % (step, json.dumps(validate(VAL))))
print("[DMD_DONE] %d steps, %.0f min" % (step, (time.time() - t0) / 60), flush=True)
