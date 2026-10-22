"""[10/16 stage 4] DMD re-distillation of the 14B causal-fast AT ITS 2-STEP SCHEDULE (nodes 0,500 = the rt2 draft, sigma 1.0 -> 0.833 -> 0),
Self-Forcing style (paper/meeting_1012/stage4_design.md). Derived from specwm/dmd_any.py (1.3B pilot, facts §34) with the flow-map parts removed:
  generator  = 14B causal-fast + LoRA 'gen' (rank --rank, q/k/v/o/ffn), free-running rollouts in its own KV (no exposure bias);
               per chunk a random stop step k in {1..K}: the chunk's x0 estimate at step k is the sample AND the committed history (Self-Forcing);
  real score = the bidirectional 14B (lingbot-world-v2/bid_as_v1_cam, two experts, CFG --guide) on cuda:1, forward only (dmd_any.real_x0);
  fake score = the same causal-fast base + LoRA 'fake', flow matching on fresh generator rollouts (teacher-forced on the generated clip);
  DMD grad   = (x0_fake - x0_real) / mean|x - x0_real| at sigma ~ shift-5 U[smin, smax] on the whole clip; backprop through the stop step only,
               --grad-chunks random chunks per step (backward per chunk, history re-committed without grad).
Saves <out>/lora_s<step>.pt / lora_last.pt in specwm.lora_util.state format (loads into two_machine --student) and fake_last.pt (critic).
usage: CUDA_VISIBLE_DEVICES=g0,g1 python -m specwm.dmd14 --out specwm_out/dmd14/R1 [--lr-gen 2e-6 --rank 64 --steps 2000 --frames 81]"""
import argparse, os, sys, glob, json, time, math, random, copy, importlib, subprocess, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--frames", type=int, default=81); ap.add_argument("--area", type=int, default=270336)
ap.add_argument("--nodes", default="0,500", help="student schedule (timesteps_index of the shift-5 scheduler); 0,500 = rt2"); ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--critic-ratio", type=int, default=2); ap.add_argument("--lr-gen", type=float, default=2e-6); ap.add_argument("--lr-fake", type=float, default=2e-5)
ap.add_argument("--rank", type=int, default=64); ap.add_argument("--ffn", type=int, default=1); ap.add_argument("--guide", type=float, default=5.0)
ap.add_argument("--grad-chunks", type=int, default=3); ap.add_argument("--smin", type=float, default=0.02); ap.add_argument("--smax", type=float, default=0.98); ap.add_argument("--shift", type=float, default=5.0)
ap.add_argument("--p-stop", default="", help="probabilities of stop step k=1..K (default uniform)"); ap.add_argument("--save-every", type=int, default=100); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--log-every", type=int, default=10); ap.add_argument("--resume", default="", help="dir with lora_last.pt + fake_last.pt + state.json")
ap.add_argument("--streams", default="task,tau1_s0,tau1_s1,tau1_s2,tau1_s3,tau2_s0,tau2_s1,tau2_s2,tau2_s3,tau2j03_s0,tau2j03_s1,tau2j1_s0,tau2j1_s1,tau4_s0,tau4_s1,tau4_s2,tau4_s3", help="task = the task's own action file")
ap.add_argument("--tasks", default=S + "/lingbot-world-v2/specwm_out/refiner/train_tasks.txt", help="47 non-bench tasks (scene-disjoint from bench-43)")
ap.add_argument("--sanity", type=int, default=1, help="decode the first step's generated clip and the bid's x0_real to <out>/sanity_*.mp4"); ap.add_argument("--smoke", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); random.seed(a.seed); torch.manual_seed(a.seed)
dev = torch.device("cuda:0"); dev1 = torch.device("cuda:1"); CS = 3; NODES = tuple(int(v) for v in a.nodes.split(",")); K = len(NODES)
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding
from specwm import lora_util as LU
from wan.utils.cam_utils import get_Ks_transformed, get_plucker_embeddings
from einops import rearrange
FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"; TT_ = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
b = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=NODES, shift=a.shift)
_te = b.pipe.text_encoder
class _TE:
    def __init__(s, te): s.te = te; s.model = te.model
    def __call__(s, texts, device): return s.te(texts, dev)
b.pipe.text_encoder = _TE(_te); m = b.target_model.eval(); SIG = [float(s) for s in b.node_sigmas()]
for p in m.parameters(): p.requires_grad_(False)
# ---- two gated adapters on every q,k,v,o,ffn0,ffn2: 'gen' (GG) and 'fake' (GF); gen is saved in lora_util.state layout (block order, same linears) ----
GG = torch.ones((), device=dev); GF = torch.zeros((), device=dev)
class DLoRA(nn.Module):
    def __init__(s, base, r):
        super().__init__(); fi, fo = base.in_features, base.out_features; s.base = base
        s.A = nn.Parameter(torch.randn(r, fi, device=dev) / math.sqrt(fi)); s.B = nn.Parameter(torch.zeros(fo, r, device=dev))
        s.A2 = nn.Parameter(torch.randn(r, fi, device=dev) / math.sqrt(fi)); s.B2 = nn.Parameter(torch.zeros(fo, r, device=dev))
    def forward(s, x):
        xf = x.float(); return s.base(x) + (GG * ((xf @ s.A.t()) @ s.B.t()) + GF * ((xf @ s.A2.t()) @ s.B2.t())).to(x.dtype)
LOR = []
for blk in m.blocks:
    for nm in ("q", "k", "v", "o"): l = DLoRA(getattr(blk.self_attn, nm), a.rank); setattr(blk.self_attn, nm, l); LOR.append(l)
    if a.ffn:
        for i in (0, 2): l = DLoRA(blk.ffn[i], a.rank); blk.ffn[i] = l; LOR.append(l)
P_GEN = [p for l in LOR for p in (l.A, l.B)]; P_FAKE = [p for l in LOR for p in (l.A2, l.B2)]
def adapter(which): GG.fill_(1.0 if which == "gen" else 0.0); GF.fill_(1.0 if which == "fake" else 0.0)
def save_gen(step, tag): torch.save(dict(rank=a.rank, ffn=a.ffn, skip=[], mode="dmd", nodes=list(NODES), step=step, A=[l.A.detach().cpu() for l in LOR], B=[l.B.detach().cpu() for l in LOR]), a.out + "/lora%s.pt" % tag)
def save_fake(step): torch.save(dict(step=step, A2=[l.A2.detach().cpu() for l in LOR], B2=[l.B2.detach().cpu() for l in LOR]), a.out + "/fake_last.pt")
from torch.utils.checkpoint import checkpoint
for blk in m.blocks:
    _f = blk.forward
    def _ck(*args, _f=_f, **kw): return checkpoint(_f, *args, use_reentrant=False, **kw) if torch.is_grad_enabled() else _f(*args, **kw)
    blk.forward = _ck
opt_g = torch.optim.AdamW(P_GEN, lr=a.lr_gen, weight_decay=0.0, betas=(0.0, 0.999)); opt_f = torch.optim.AdamW(P_FAKE, lr=a.lr_fake, weight_decay=0.0, betas=(0.0, 0.999))
step = 0
if a.resume:
    ck = torch.load(a.resume + "/lora_last.pt", map_location="cpu"); ckf = torch.load(a.resume + "/fake_last.pt", map_location="cpu")
    for l, A_, B_, A2_, B2_ in zip(LOR, ck["A"], ck["B"], ckf["A2"], ckf["B2"]): l.A.data.copy_(A_.to(dev)); l.B.data.copy_(B_.to(dev)); l.A2.data.copy_(A2_.to(dev)); l.B2.data.copy_(B2_.to(dev))
    step = int(ck["step"]); print("[dmd14] resumed %s at step %d" % (a.resume, step), flush=True)
def shift_sig(s): return a.shift * s / (1 + (a.shift - 1) * s)
def sample_sig(): return float(min(max(shift_sig(random.uniform(0.0, 1.0)), a.smin), a.smax))
PSTOP = [float(v) for v in a.p_stop.split(",")] if a.p_stop else [1.0 / K] * K; assert len(PSTOP) == K
def sample_stop(): return random.choices(range(1, K + 1), weights=PSTOP)[0]
def gen(seed, c, k=0): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k * 100003)
# ---- conditioning: the task's own action (stream 'task') or a synthetic camera stream (data/actstream, as in the pilot) ----
def emb_frames(rel, Ks, h, w, lat_h, lat_w):
    f = rel.shape[0]; e = get_plucker_embeddings(rel, Ks, h, w); e = rearrange(e, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)', c1=int(h // lat_h), c2=int(w // lat_w))[None, ...]
    return rearrange(e, 'b (f h w) c -> b c f h w', f=f, h=lat_h, w=lat_w).to(b.param_dtype)
def stream_cam(stream, N, h, w, lat_h, lat_w):
    rel = torch.from_numpy(np.load(S + "/data/actstream/%s/rel.npy" % stream)).float().to(dev)[:N * CS]
    Ks = get_Ks_transformed(torch.from_numpy(np.load(S + "/data/actstream/%s/intrinsics.npy" % stream)).float(), height_org=480, width_org=832, height_resize=h, width_resize=w, height_final=h, width_final=w)[0].repeat(N * CS, 1).to(dev)
    return rel, Ks
COND = {}; _CK = ("_ctx", "_y_chunks", "_emb_chunks", "_chunk_shape", "_frame_seqlen", "_chunk_size", "_kv_size", "_seq_len", "_mk_cache")
def setup_task(task, stream):
    """student conditioning for (task, stream); returns (Nc, full-clip plucker [1,C,F,h,w] for the bid). The prepared conditioning (text ctx,
    image latent, plucker of the task's own action, cache maker) is cached per task so the T5 / VAE run once per task."""
    T = TT_[task]
    if task not in COND:
        _te.model.to(dev)
        with torch.no_grad(): cond = b.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=a.frames, max_area=a.area)
        _te.model.to("cpu"); torch.cuda.empty_cache(); COND[task] = dict(N=cond["num_chunks"], **{k_: getattr(b, k_) for k_ in _CK})
    for k_ in _CK: setattr(b, k_, COND[task][k_])
    b.reset_caches(); N = COND[task]["N"]; lat_h, lat_w = b._chunk_shape[2], b._chunk_shape[3]
    if stream == "task": return N, torch.cat(list(b._emb_chunks), 2)
    h, w = lat_h * 8, lat_w * 8; rel, Ks = stream_cam(stream, N, h, w, lat_h, lat_w)
    b._emb_chunks = [emb_frames(rel[CS * c:CS * c + CS], Ks[CS * c:CS * c + CS], h, w, lat_h, lat_w) for c in range(N)]
    return N, emb_frames(rel, Ks, h, w, lat_h, lat_w)
# ---- the bidirectional teacher (v1 layout) on cuda:1 (dmd_any.py pattern) ----
os.chdir(S + "/lingbot-world"); sys.path = [S + "/lingbot-world"] + [p for p in sys.path if p != S + "/lingbot-world-v2"]
for k_ in [k_ for k_ in sys.modules if k_ == "wan" or k_.startswith("wan.")]: del sys.modules[k_]
_spec = importlib.util.spec_from_file_location("specwm_v1", S + "/lingbot-world/specwm/__init__.py", submodule_search_locations=[S + "/lingbot-world/specwm"]); _pkg = importlib.util.module_from_spec(_spec); sys.modules["specwm_v1"] = _pkg; _spec.loader.exec_module(_pkg)
V1Binding = importlib.import_module("specwm_v1.binding_v1").V1Binding
b1 = V1Binding(S + "/lingbot-world-v2/bid_as_v1_cam", t5_cpu=True, device=str(dev1)); b1.device = dev1
b1.pipe.high_noise_model.to(dev1).eval(); b1.pipe.low_noise_model.to(dev1).eval()
TASKS = [t.strip() for t in open(a.tasks) if t.strip()]; STREAMS = a.streams.split(","); _te1 = b1.pipe.text_encoder; _te1.model.to(dev1); b1._txt_cache = {}
with torch.no_grad():
    _ctxn = [t.to(dev1) for t in _te1([b1.pipe.sample_neg_prompt], dev1)]
    for _tk in TASKS:
        _pr = TT_[_tk].get("camera_WM", "")
        if (_pr, b1.pipe.sample_neg_prompt) not in b1._txt_cache: b1._txt_cache[(_pr, b1.pipe.sample_neg_prompt)] = ([t.to(dev1) for t in _te1([_pr], dev1)], _ctxn)
_te1.model.cpu(); torch.cuda.empty_cache()
for mm in (b1.pipe.high_noise_model, b1.pipe.low_noise_model):
    for p in mm.parameters(): p.requires_grad_(False)
os.chdir(S + "/lingbot-world-v2"); print("[dmd14] bid 14B on %s: %.1f GiB; %d tasks x %d streams" % (dev1, torch.cuda.memory_allocated(dev1) / 2**30, len(TASKS), len(STREAMS)), flush=True)
BID_COND = {}
def bid_cond(task, plucker):
    key = task
    if key not in BID_COND:
        T = TT_[task]
        with torch.no_grad(): c = b1.prepare(T["image"], T.get("camera_WM", ""), None, frame_num=a.frames, max_area=a.area)
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
# ---- generator rollout: the deployed sampler (fresh re-noise between nodes, as two_machine.denoise) with a random stop step per chunk ----
@torch.no_grad()
def rollout(Nc, seed, stops):
    """stops[c] in 1..K: the chunk's x0 estimate after that many steps is its sample and its committed history.
    returns x [16, Nc*CS, h, w], zin[c] = the latent entering the stop step"""
    adapter("gen"); b.reset_caches(); g = gen(seed, 0); xs = []; zin = []
    for c in range(Nc):
        x = None
        for j in range(stops[c]):
            if j == 0: z = b.chunk_noise(c, g)
            else: s_ = SIG[j]; z = (1 - s_) * x + s_ * b.randn_like(x, g)
            if j == stops[c] - 1: zin.append(z.clone())
            x = b.target_x0(z, j, c).float()
        b.commit(x, c); xs.append(x)
    return torch.cat(xs, 1), zin
def fake_x0_clip(z, x, sig, grad=False, loss_tgt=None):
    """critic x0 for the noisy clip z, teacher-forced on the clean clip x (its own causal context). grad: flow-matching loss to loss_tgt per chunk
    (backward per chunk) and the per-chunk losses are returned instead of x0"""
    adapter("fake"); b.reset_caches(); Nc = x.shape[1] // CS; outs = []; losses = []
    for c in range(Nc):
        zc = z[:, CS * c:CS * c + CS]
        with torch.set_grad_enabled(grad):
            u = b.target_flow(zc, float(sig), c)
            if grad: l = Fn.mse_loss(u, loss_tgt[:, CS * c:CS * c + CS]); (l / Nc).backward(); losses.append(float(l))
            else: outs.append((zc - float(sig) * u).detach())
        with torch.no_grad(): b.commit(x[:, CS * c:CS * c + CS], c)
    adapter("gen"); return (torch.cat(outs, 1) if outs else None), losses
def logw(msg):
    print("[dmd14 %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True); open(a.out + "/log.txt", "a").write(msg + "\n")
def pick(): return random.choice(TASKS), random.choice(STREAMS), random.randrange(1 << 30)
H = dict(dmd=[], gn=[], fake=[], sig=[]); t0 = time.time(); TM = dict(roll=0.0, bid=0.0, fake=0.0, replay=0.0, critic=0.0)
logw("start step %d: nodes %s sigmas %s, LoRA r%d %.0fM params x2, lr gen %g fake %g, critic ratio %d, grad chunks %d, frames %d area %d" % (step, NODES, [round(s, 3) for s in SIG], a.rank, sum(p.numel() for p in P_GEN) / 1e6, a.lr_gen, a.lr_fake, a.critic_ratio, a.grad_chunks, a.frames, a.area))
if a.smoke: a.steps = step + a.smoke
while step < a.steps:
    # ---- critic steps: flow matching of the fake score on fresh generator rollouts ----
    for _ in range(a.critic_ratio):
        _t = time.time(); task, stream, seed = pick(); Nc, pl = setup_task(task, stream); stops = [sample_stop() for _ in range(Nc)]
        x, _ = rollout(Nc, seed, stops); sig = sample_sig(); eps = torch.randn_like(x); z = (1 - sig) * x + sig * eps
        _, ls = fake_x0_clip(z, x, sig, grad=True, loss_tgt=eps - x)
        torch.nn.utils.clip_grad_norm_(P_FAKE, 1.0); opt_f.step(); opt_f.zero_grad(set_to_none=True); H["fake"].append(float(np.mean(ls))); TM["critic"] += time.time() - _t
    # ---- generator step ----
    _t = time.time(); task, stream, seed = pick(); Nc, pl = setup_task(task, stream); stops = [sample_stop() for _ in range(Nc)]
    x, zin = rollout(Nc, seed, stops); torch.cuda.synchronize(); TM["roll"] += time.time() - _t
    _t = time.time(); c1 = bid_cond(task, pl); sig = sample_sig(); eps = torch.randn_like(x); z = (1 - sig) * x + sig * eps
    x0_real = real_x0(z, sig, c1); torch.cuda.synchronize(); TM["bid"] += time.time() - _t
    _t = time.time(); x0_fake, _ = fake_x0_clip(z, x, sig); torch.cuda.synchronize(); TM["fake"] += time.time() - _t
    with torch.no_grad():
        nrm = (x - x0_real).abs().mean().clamp_min(1e-6); gvec = torch.nan_to_num((x0_fake - x0_real) / nrm); tgt = x - gvec
    H["gn"].append(float(gvec.norm() / x.norm())); H["sig"].append(sig)
    if a.sanity and step == 0 and not os.path.exists(a.out + "/sanity_gen.mp4"):
        with torch.no_grad():
            for nm_, lat in (("gen", x), ("real", x0_real), ("fake", x0_fake)):
                os.makedirs(a.out + "/sanity_" + nm_, exist_ok=True); b.decode_and_save([lat[:, CS * c:CS * c + CS] for c in range(Nc)], a.out + "/sanity_" + nm_)
                if os.path.exists(a.out + "/sanity_%s/spec.mp4" % nm_): os.replace(a.out + "/sanity_%s/spec.mp4" % nm_, a.out + "/sanity_%s.mp4" % nm_)
        logw("sanity decodes written (task %s stream %s sigma %.2f stops %s)" % (task, stream, sig, stops)); torch.cuda.empty_cache()
    # replay the stop step of `grad-chunks` chunks with grad (history re-committed from x without grad)
    _t = time.time(); adapter("gen"); b.reset_caches(); G = set(random.sample(range(Nc), min(a.grad_chunks, Nc))); dl = []
    for c in range(Nc):
        if c in G:
            xc = b.target_x0(zin[c], stops[c] - 1, c).float(); l = 0.5 * Fn.mse_loss(xc, tgt[:, CS * c:CS * c + CS]); (l / len(G)).backward(); dl.append(float(l))
        with torch.no_grad(): b.commit(x[:, CS * c:CS * c + CS], c)
    H["dmd"].append(float(np.mean(dl))); torch.cuda.synchronize(); TM["replay"] += time.time() - _t
    torch.nn.utils.clip_grad_norm_(P_GEN, 1.0); opt_g.step(); opt_g.zero_grad(set_to_none=True); step += 1
    if step % a.log_every == 0 or a.smoke:
        logw("step %d  sig %.2f stops %s  dmd %.4f |grad|/|x| %.3f  fake %.4f  %.0f min  %.1f/%.1f GiB  t(roll/bid/fake/replay/critic) %.0f/%.0f/%.0f/%.0f/%.0f s" % (
            step, sig, "".join(str(s_) for s_ in stops), np.mean(H["dmd"][-a.log_every:]), np.mean(H["gn"][-a.log_every:]), np.mean(H["fake"][-a.log_every * a.critic_ratio:] or [0]), (time.time() - t0) / 60,
            torch.cuda.max_memory_allocated(dev) / 2**30, torch.cuda.max_memory_allocated(dev1) / 2**30, TM["roll"], TM["bid"], TM["fake"], TM["replay"], TM["critic"]))
    if step % a.save_every == 0 or step >= a.steps:
        save_gen(step, "_s%d" % step); save_gen(step, "_last"); save_fake(step); json.dump(dict(step=step, H={k_: v[-2000:] for k_, v in H.items()}), open(a.out + "/state.json", "w"))
print("[DMD14_DONE] %d steps, %.0f min" % (step, (time.time() - t0) / 60), flush=True)
