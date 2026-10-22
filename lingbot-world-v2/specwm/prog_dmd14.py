"""[10/22] PROGRESSIVE 2-step device student by Self-Forcing DMD (user's framing 10/22; design paper/1026/progressive_student_design.md).
Student sampler (= the deployed one): chunk c -> step 1 at the HALF latent grid (sigma 1.0, own pooled KV) -> x0_lo -> bilinear up to the full grid
-> re-noise to the second node (sigma 0.833) -> step 2 at FULL resolution (full KV) -> x0 -> commit to the full KV and (down-sampled) to the half KV.
Derived from specwm/dmd14.py (2-step full-res DMD, never run before 10/22) + specwm/multires_pilot.py (two binding states HI / LO, pooled noise,
up / down):  generator = 14B causal-fast + adapters (--adapt lora: two LoRAs 'lo' (half-res step) and 'hi' (full-res step), rank --rank, q/k/v/o/ffn;
             --adapt full: the whole DiT is trained, both steps share the weights and only the resolution differs -- the quality-first setting for
             the cluster; the critic is then a LoRA on a frozen copy of the base = rank 'fake'), free-running rollouts in the device config
             (--window, --kv fp8 optional) with a random stop step k in {1, 2} per chunk (k = 1: the sample is up(x0_lo), k = 2: the finished chunk);
  real score = the bidirectional 14B (bid_as_v1_cam, two experts, CFG --guide) on the SECOND GPU, forward only (dmd14.real_x0);
  fake score = the causal-fast base + LoRA 'fake', flow matching on fresh rollouts, teacher-forced on the generated clip at full res;
  DMD grad   = (x0_fake - x0_real) / mean|x - x0_real| at sigma ~ shift-5 U[smin, smax] on the whole clip; backprop through the stop step only
               (k = 2: the full-res step; k = 1: the half-res step through up()), --grad-chunks chunks per step.
  --prog 0  = plain full-res 2-step student (= dmd14) with the SAME code path, for the "cost of progressive" comparison run.
Roles / parallelism (single script): --roles gen,real = this process holds the generator on cuda:0 and the bid on cuda:1 (local 2-GPU smoke, as dmd14);
  --dist 1 (torchrun, one GPU per rank): rank 0..R-2 = generator replicas (DDP: gradients all-reduced, batch = R-1 rollouts per step), rank R-1 = real-score
  server (receives (z, sigma, cond id) by torch.distributed point-to-point, returns x0_real). Full fine-tune on 1-GPU nodes (Vista GH200 95 GB): bf16 weights
  + bf16 grads + 8-bit Adam (--opt adam8) + chunk checkpointing; on 4-GPU nodes (Stampede3 H100) use --fsdp 1 for the generator group instead of DDP.
Saves <out>/student_s<step>.pt / student_last.pt: {"adapt": "lora"|"full", "prog": 1, "nodes": [...], "lo": LoRA state (lora_util layout), "hi": LoRA state}
or the full DiT state dict (adapt full), loadable by two_machine --student (progressive flag read from the file; the loader is in specwm/prog_student.py).
usage (local smoke):  CUDA_VISIBLE_DEVICES=g0,g1 python -m specwm.prog_dmd14 --out specwm_out/prog_dmd14/S0 --adapt lora --smoke 3
       (cluster)     torchrun --nnodes 8 --nproc_per_node 1 ... -m specwm.prog_dmd14 --out ... --adapt full --dist 1 --opt adam8 --steps 5000"""
import argparse, os, sys, glob, json, time, math, random, copy, importlib, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
S = os.environ.get("SPECWM_ROOT", "/home/hk28965/specwm")
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--frames", type=int, default=81); ap.add_argument("--area", type=int, default=270336, help="full-res max_area (832x480=399360, 704x384=270336, 640x352=225281); --areas overrides with a list sampled per step")
ap.add_argument("--areas", default="", help="comma list of max_areas sampled uniformly per step (resolution mix), e.g. 225281,399360")
ap.add_argument("--nodes", default="0,500", help="student schedule (timesteps_index of the shift-5 scheduler); 0,500 = the rt2 2-step"); ap.add_argument("--prog", type=int, default=1, help="1 = step 1 at the half grid; 0 = plain full-res (dmd14)")
ap.add_argument("--adapt", default="lora", choices=["lora", "full"]); ap.add_argument("--rank", type=int, default=64); ap.add_argument("--ffn", type=int, default=1); ap.add_argument("--rank-fake", type=int, default=64)
ap.add_argument("--window", type=int, default=12, help="device KV window (local_attn) during rollouts = the deployed squeeze"); ap.add_argument("--kv", default="bf16", choices=["bf16", "fp8"], help="(reserved) FP8 KV ring during rollouts; bf16 for training")
ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--critic-ratio", type=int, default=2); ap.add_argument("--lr-gen", type=float, default=2e-6); ap.add_argument("--lr-fake", type=float, default=2e-5)
ap.add_argument("--opt", default="adamw", choices=["adamw", "adam8"], help="adam8 = bitsandbytes 8-bit AdamW (full fine-tune on 95-GB GPUs)"); ap.add_argument("--guide", type=float, default=5.0)
ap.add_argument("--grad-chunks", type=int, default=3); ap.add_argument("--smin", type=float, default=0.02); ap.add_argument("--smax", type=float, default=0.98); ap.add_argument("--shift", type=float, default=5.0)
ap.add_argument("--p-stop", default="", help="probabilities of stop step k=1..K (default uniform)"); ap.add_argument("--save-every", type=int, default=100); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of the trained weights (0 = off); the EMA copy is what gets saved as student_ema_*.pt")
ap.add_argument("--log-every", type=int, default=10); ap.add_argument("--resume", default="", help="dir with student_last.pt + fake_last.pt + state.json")
ap.add_argument("--streams", default="task,tau1_s0,tau1_s1,tau1_s2,tau1_s3,tau2_s0,tau2_s1,tau2_s2,tau2_s3,tau2j03_s0,tau2j03_s1,tau2j1_s0,tau2j1_s1,tau4_s0,tau4_s1,tau4_s2,tau4_s3", help="task = the task's own action file; names under data/actstream/")
ap.add_argument("--tasks", default=S + "/lingbot-world-v2/specwm_out/refiner/train_tasks.txt", help="task ids (one per line) of data/stevo/tasks_camera.json (47 non-bench tasks) or of --tasks-json")
ap.add_argument("--tasks-json", default=S + "/data/stevo/tasks_camera.json", help="task table {id, image, camera_WM (caption), action (poses dir)}; the expanded pool (gen/make_train_pool.py) is another json of the same layout")
ap.add_argument("--sanity", type=int, default=1, help="decode the first step's generated clip and the bid's x0_real to <out>/sanity_*.mp4"); ap.add_argument("--smoke", type=int, default=0)
ap.add_argument("--roles", default="gen,real", help="single process: gen,real = generator on cuda:0 + bid on cuda:1 (local)"); ap.add_argument("--dist", type=int, default=0, help="1 = torchrun multi-rank: ranks 0..R-2 generators (DDP), rank R-1 real-score server")
ap.add_argument("--fsdp", type=int, default=0, help="(cluster, 4-GPU nodes) 1 = shard the full-FT generator over the generator ranks with FSDP instead of DDP replicas")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
# ---------------------------------------------------------------- distributed roles ----------------------------------------------------------------
RANK, WORLD = 0, 1
if a.dist:
    import torch.distributed as dist
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo"); RANK, WORLD = dist.get_rank(), dist.get_world_size(); assert WORLD >= 2, "need >= 2 ranks (generators + 1 real-score server)"
    IS_REAL = RANK == WORLD - 1; GEN_RANKS = list(range(WORLD - 1)); dev = torch.device("cuda:0"); dev1 = torch.device("cuda:0")   # one GPU per rank
    GEN_GROUP = dist.new_group(GEN_RANKS); torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
else:
    IS_REAL = False; GEN_RANKS = [0]; dev = torch.device("cuda:0"); dev1 = torch.device("cuda:1")
random.seed(a.seed + RANK); torch.manual_seed(a.seed + RANK)
CS = 3; NODES = tuple(int(v) for v in a.nodes.split(",")); K = len(NODES); assert K == 2, "progressive student = exactly 2 nodes (half-res step, full-res step)"
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from wan.utils.cam_utils import get_Ks_transformed, get_plucker_embeddings
from einops import rearrange
TT_ = {t["id"]: t for t in json.load(open(a.tasks_json))}; TASKS = [t.strip() for t in open(a.tasks) if t.strip()]; STREAMS = a.streams.split(",")
AREAS = [int(v) for v in a.areas.split(",")] if a.areas else [a.area]
def logw(msg):
    if RANK in (0, WORLD - 1) or not a.dist:
        print("[prog_dmd14 r%d %s] %s" % (RANK, time.strftime("%H:%M:%S"), msg), flush=True)
        if RANK == 0: open(a.out + "/log.txt", "a").write(msg + "\n")
# ================================================================ real-score side (the bidirectional 14B, v1 layout) ================================
def build_bid(device):
    os.chdir(S + "/lingbot-world"); sys.path = [S + "/lingbot-world"] + [p for p in sys.path if p != S + "/lingbot-world-v2"]
    for k_ in [k_ for k_ in sys.modules if k_ == "wan" or k_.startswith("wan.")]: del sys.modules[k_]
    _spec = importlib.util.spec_from_file_location("specwm_v1", S + "/lingbot-world/specwm/__init__.py", submodule_search_locations=[S + "/lingbot-world/specwm"]); _pkg = importlib.util.module_from_spec(_spec); sys.modules["specwm_v1"] = _pkg; _spec.loader.exec_module(_pkg)
    V1Binding = importlib.import_module("specwm_v1.binding_v1").V1Binding
    b1 = V1Binding(S + "/lingbot-world-v2/bid_as_v1_cam", t5_cpu=True, device=str(device)); b1.device = device
    b1.pipe.high_noise_model.to(device).eval(); b1.pipe.low_noise_model.to(device).eval(); _te1 = b1.pipe.text_encoder; _te1.model.to(device); b1._txt_cache = {}
    with torch.no_grad():
        _ctxn = [t.to(device) for t in _te1([b1.pipe.sample_neg_prompt], device)]
        for _tk in TASKS:
            _pr = TT_[_tk].get("camera_WM", "")
            if (_pr, b1.pipe.sample_neg_prompt) not in b1._txt_cache: b1._txt_cache[(_pr, b1.pipe.sample_neg_prompt)] = ([t.to(device) for t in _te1([_pr], device)], _ctxn)
    _te1.model.cpu(); torch.cuda.empty_cache()
    for mm in (b1.pipe.high_noise_model, b1.pipe.low_noise_model):
        for p in mm.parameters(): p.requires_grad_(False)
    os.chdir(S + "/lingbot-world-v2"); logw("bid 14B on %s: %.1f GiB; %d tasks x %d streams" % (device, torch.cuda.memory_allocated(device) / 2**30, len(TASKS), len(STREAMS))); return b1
BID_COND = {}
def bid_cond(b1, task, area, plucker):
    key = (task, area)
    if key not in BID_COND:
        T = TT_[task]
        with torch.no_grad(): c = b1.prepare(T["image"], T.get("camera_WM", ""), None, frame_num=a.frames, max_area=area)
        BID_COND[key] = c
    c = dict(BID_COND[key]); lat_f = c["lat"][0]; assert plucker.shape[2] == lat_f, (plucker.shape, c["lat"])
    c["dit_cond"] = {"c2ws_plucker_emb": plucker.to(b1.device, b1.param_dtype).chunk(1, dim=0)}; return c
@torch.no_grad()
def real_x0(b1, z, sig, c):
    """x0 prediction of the bidirectional teacher (CFG, expert by noise level) for the whole clip z [16,F,h,w] at sigma"""
    d1 = b1.device; t_val = float(sig) * 1000.0; mm = b1.pipe.high_noise_model if t_val >= b1.boundary_t else b1.pipe.low_noise_model
    tt = torch.full((1,), t_val, device=d1); zin = [z.to(d1, b1.param_dtype)]
    with torch.amp.autocast("cuda", dtype=b1.param_dtype):
        fc = mm(zin, t=tt, context=[c["ctx"][0]], seq_len=c["seq_len"], y=[c["y"]], dit_cond_dict=c["dit_cond"])[0]
        fu = mm(zin, t=tt, context=c["ctxn"], seq_len=c["seq_len"], y=[c["y"]], dit_cond_dict=c["dit_cond"])[0]
    flow = (fu + a.guide * (fc - fu)).float(); return (z.to(d1).float() - float(sig) * flow)
# ---- real-score SERVER loop (dist rank R-1): serves the generator ranks round-robin until a stop message ----
# protocol (per request): header long tensor [task_idx, area, F, h, w, stop_flag, sig*1e6, C] then z [16,F,h,w] fp32 and plucker [1,C,F,h,w] fp32; reply x0_real fp32
if a.dist and IS_REAL:
    b1 = build_bid(dev1)
    while True:
        for src in GEN_RANKS:
            hdr = torch.zeros(8, dtype=torch.long); dist.recv(hdr, src=src)
            if int(hdr[5]) == 1: logw("real-score server: stop from rank %d" % src); dist.barrier(); sys.exit(0)
            ti, area, F_, h, w = [int(v) for v in hdr[:5]]; sig = float(hdr[6]) / 1e6; C_ = int(hdr[7])
            z = torch.zeros(16, F_, h, w); dist.recv(z, src=src); pl = torch.zeros(1, C_, F_, h, w); dist.recv(pl, src=src)
            x0 = real_x0(b1, z.to(dev1), sig, bid_cond(b1, TASKS[ti], area, pl.to(dev1))).cpu(); dist.send(x0.contiguous(), dst=src)
# ================================================================ generator side ======================================================================
from specwm.binding_v2 import V2Binding
from specwm import lora_util as LU
FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
b = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=a.window, swap_models=False, build_draft=False, timesteps_index=NODES, shift=a.shift)
_te = b.pipe.text_encoder
class _TE:
    def __init__(s, te): s.te = te; s.model = te.model
    def __call__(s, texts, device): return s.te(texts, dev)
b.pipe.text_encoder = _TE(_te); m = b.target_model.eval(); SIG = [float(s) for s in b.node_sigmas()]
for p in m.parameters(): p.requires_grad_(False)
# ---- adapters: GL (lo step), GH (hi step), GF (fake critic) gates on every q,k,v,o[,ffn0,ffn2]; adapt full = the base weights train and GL/GH LoRAs are absent ----
GL = torch.zeros((), device=dev); GH = torch.ones((), device=dev); GF = torch.zeros((), device=dev)
class TriLoRA(nn.Module):
    def __init__(s, base, r, rf, lo_hi):
        super().__init__(); fi, fo = base.in_features, base.out_features; s.base = base; s.lo_hi = lo_hi
        if lo_hi:
            s.A = nn.Parameter(torch.randn(r, fi, device=dev) / math.sqrt(fi)); s.B = nn.Parameter(torch.zeros(fo, r, device=dev))        # lo
            s.A3 = nn.Parameter(torch.randn(r, fi, device=dev) / math.sqrt(fi)); s.B3 = nn.Parameter(torch.zeros(fo, r, device=dev))      # hi
        s.A2 = nn.Parameter(torch.randn(rf, fi, device=dev) / math.sqrt(fi)); s.B2 = nn.Parameter(torch.zeros(fo, rf, device=dev))        # fake critic
    def forward(s, x):
        xf = x.float(); out = s.base(x); d = GF * ((xf @ s.A2.t()) @ s.B2.t())
        if s.lo_hi: d = d + GL * ((xf @ s.A.t()) @ s.B.t()) + GH * ((xf @ s.A3.t()) @ s.B3.t())
        return out + d.to(x.dtype)
LOR = []; lo_hi = a.adapt == "lora"
for blk in m.blocks:
    for nm in ("q", "k", "v", "o"): l = TriLoRA(getattr(blk.self_attn, nm), a.rank, a.rank_fake, lo_hi); setattr(blk.self_attn, nm, l); LOR.append(l)
    if a.ffn:
        for i in (0, 2): l = TriLoRA(blk.ffn[i], a.rank, a.rank_fake, lo_hi); blk.ffn[i] = l; LOR.append(l)
P_FAKE = [p for l in LOR for p in (l.A2, l.B2)]
if a.adapt == "lora": P_GEN = [p for l in LOR for p in (l.A, l.B, l.A3, l.B3)]
else:
    P_GEN = [p for n_, p in m.named_parameters() if "A2" not in n_ and "B2" not in n_]
    for p in P_GEN: p.requires_grad_(True)
    if a.fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        for i, blk in enumerate(m.blocks): m.blocks[i] = FSDP(blk, process_group=GEN_GROUP, use_orig_params=True)
def adapter(which, step_lo=False):
    """which = gen | fake; step_lo = the half-res step (lo LoRA) vs the full-res step (hi LoRA); with adapt full both steps use the trained base"""
    GF.fill_(1.0 if which == "fake" else 0.0); on = which == "gen" and lo_hi; GL.fill_(1.0 if on and step_lo else 0.0); GH.fill_(1.0 if on and not step_lo else 0.0)
def student_state(step):
    if a.adapt == "lora": return dict(adapt="lora", prog=a.prog, rank=a.rank, ffn=a.ffn, skip=[], mode="prog_dmd", nodes=list(NODES), window=a.window, step=step,
                                     lo=dict(A=[l.A.detach().cpu() for l in LOR], B=[l.B.detach().cpu() for l in LOR]), hi=dict(A=[l.A3.detach().cpu() for l in LOR], B=[l.B3.detach().cpu() for l in LOR]))
    sd = {k_: v.detach().cpu() for k_, v in m.state_dict().items() if "A2" not in k_ and "B2" not in k_}
    return dict(adapt="full", prog=a.prog, nodes=list(NODES), window=a.window, step=step, state=sd)
def save_student(step, tag): torch.save(student_state(step), a.out + "/student%s.pt" % tag)
def save_fake(step): torch.save(dict(step=step, rank=a.rank_fake, A2=[l.A2.detach().cpu() for l in LOR], B2=[l.B2.detach().cpu() for l in LOR]), a.out + "/fake_last.pt")
from torch.utils.checkpoint import checkpoint
for blk in m.blocks:
    _f = blk.forward
    def _ck(*args, _f=_f, **kw): return checkpoint(_f, *args, use_reentrant=False, **kw) if torch.is_grad_enabled() else _f(*args, **kw)
    blk.forward = _ck
if a.opt == "adam8":
    import bitsandbytes as bnb; opt_g = bnb.optim.AdamW8bit(P_GEN, lr=a.lr_gen, weight_decay=0.0, betas=(0.0, 0.999))
else: opt_g = torch.optim.AdamW(P_GEN, lr=a.lr_gen, weight_decay=0.0, betas=(0.0, 0.999))
opt_f = torch.optim.AdamW(P_FAKE, lr=a.lr_fake, weight_decay=0.0, betas=(0.0, 0.999))
EMA = [p.detach().clone() for p in P_GEN] if a.ema > 0 else None
step = 0
if a.resume:
    ck = torch.load(a.resume + "/student_last.pt", map_location="cpu"); ckf = torch.load(a.resume + "/fake_last.pt", map_location="cpu")
    if ck["adapt"] == "lora":
        for l, A_, B_, A3_, B3_ in zip(LOR, ck["lo"]["A"], ck["lo"]["B"], ck["hi"]["A"], ck["hi"]["B"]): l.A.data.copy_(A_.to(dev)); l.B.data.copy_(B_.to(dev)); l.A3.data.copy_(A3_.to(dev)); l.B3.data.copy_(B3_.to(dev))
    else: m.load_state_dict({k_: v for k_, v in ck["state"].items()}, strict=False)
    for l, A2_, B2_ in zip(LOR, ckf["A2"], ckf["B2"]): l.A2.data.copy_(A2_.to(dev)); l.B2.data.copy_(B2_.to(dev))
    step = int(ck["step"]); logw("resumed %s at step %d" % (a.resume, step))
def shift_sig(s): return a.shift * s / (1 + (a.shift - 1) * s)
def sample_sig(): return float(min(max(shift_sig(random.uniform(0.0, 1.0)), a.smin), a.smax))
PSTOP = [float(v) for v in a.p_stop.split(",")] if a.p_stop else [1.0 / K] * K; assert len(PSTOP) == K
def sample_stop(): return random.choices(range(1, K + 1), weights=PSTOP)[0]
def gen(seed, c, k=0): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k * 100003)
# ---- conditioning at TWO grids per (task, area): HI = full grid, LO = half grid (multires_pilot.prepare_res) + pooled plucker for the LO state ----
def emb_frames(rel, Ks, h, w, lat_h, lat_w):
    f = rel.shape[0]; e = get_plucker_embeddings(rel, Ks, h, w); e = rearrange(e, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)', c1=int(h // lat_h), c2=int(w // lat_w))[None, ...]
    return rearrange(e, 'b (f h w) c -> b c f h w', f=f, h=lat_h, w=lat_w).to(b.param_dtype)
def stream_cam(stream, N, h, w, lat_h, lat_w):
    rel = torch.from_numpy(np.load(S + "/data/actstream/%s/rel.npy" % stream)).float().to(dev)[:N * CS]
    Ks = get_Ks_transformed(torch.from_numpy(np.load(S + "/data/actstream/%s/intrinsics.npy" % stream)).float(), height_org=480, width_org=832, height_resize=h, width_resize=w, height_final=h, width_final=w)[0].repeat(N * CS, 1).to(dev)
    return rel, Ks
STATE_KEYS = ["_y_chunks", "_emb_chunks", "_ctx", "_seq_len", "_chunk_shape", "_frame_seqlen", "_chunk_size", "_kv_size", "_mk_cache", "_caches", "_cross_init", "_committed_frames"]
def snapshot(): return {k: getattr(b, k) for k in STATE_KEYS}
def use(st):
    for k, v in st.items(): setattr(b, k, v)
def area_for(image, lh, lw):
    from PIL import Image; im = Image.open(image); asp = im.height / im.width; nom = 64 * lh * lw
    for A in range(int(nom * 0.6), int(nom * 1.7), 32):
        if (round(np.sqrt(A * asp) // 8 // 2 * 2), round(np.sqrt(A / asp) // 8 // 2 * 2)) == (lh, lw): return A
    raise RuntimeError("no max_area gives grid %dx%d for aspect %.4f" % (lh, lw, asp))
def pool_mat(Hf, hl):
    bb = np.round(np.linspace(0, Hf, hl + 1)).astype(int); P = torch.zeros(hl, Hf)
    for i in range(hl): P[i, bb[i]:bb[i + 1]] = 1.0 / np.sqrt(bb[i + 1] - bb[i])
    return P.to(dev)
def pool_noise(e, Ph, Pw): return torch.einsum("ah,cthw,bw->ctab", Ph, e, Pw)
def up(x, hw): return Fn.interpolate(x.permute(1, 0, 2, 3), size=tuple(hw), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
def down(x, hw): return Fn.interpolate(x.permute(1, 0, 2, 3), size=tuple(hw), mode="bilinear", align_corners=False, antialias=True).permute(1, 0, 2, 3)
COND = {}
def prepare_grid(T, area_or_grid):
    """one binding state for the task at the given max_area (int) or latent grid (lh, lw); caches are NOT allocated here (reset per rollout)"""
    b._caches = {}; torch.cuda.empty_cache()
    with torch.no_grad(): cond = b.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=a.frames, max_area=area_or_grid if isinstance(area_or_grid, int) else area_for(T["image"], *area_or_grid), chunk_size=CS)
    st = snapshot(); st["_caches"] = None; return st, cond["num_chunks"], tuple(cond["shape"][2:])
def setup_task(task, stream, area):
    """(Nc, full-clip plucker for the bid [1,C,F,h,w]); sets globals HI, LO (binding states), HW, LW, PH, PW. prepared once per (task, area), stream plucker per call."""
    global HI, LO, HW, LW, PH, PW
    T = TT_[task]; key = (task, area)
    if key not in COND:
        _te.model.to(dev); HI_, N, hw = prepare_grid(T, area); lw_ = (round(hw[0] / 2 / 2) * 2, round(hw[1] / 2 / 2) * 2)
        LO_, N2, hw2 = (prepare_grid(T, lw_) if a.prog else (None, N, None)); _te.model.to("cpu"); torch.cuda.empty_cache(); assert N2 == N
        COND[key] = dict(N=N, HI=HI_, LO=LO_, HW=hw, LW=lw_)
    C = COND[key]; HI, LO, HW, LW = C["HI"], C["LO"], C["HW"], C["LW"]; N = C["N"]
    if a.prog: PH, PW = pool_mat(HW[0], LW[0]), pool_mat(HW[1], LW[1])
    lat_h, lat_w = HW; h, w = lat_h * 8, lat_w * 8
    if stream == "task": pl = torch.cat(list(HI["_emb_chunks"]), 2)
    else:
        rel, Ks = stream_cam(stream, N, h, w, lat_h, lat_w); HI["_emb_chunks"] = [emb_frames(rel[CS * c:CS * c + CS], Ks[CS * c:CS * c + CS], h, w, lat_h, lat_w) for c in range(N)]
        if a.prog: LO["_emb_chunks"] = [emb_frames(rel[CS * c:CS * c + CS], Ks[CS * c:CS * c + CS], LW[0] * 8, LW[1] * 8, LW[0], LW[1]) for c in range(N)]
        pl = emb_frames(rel, Ks, h, w, lat_h, lat_w)
    return N, pl
def reset_states():
    """fresh KV caches for HI (and LO): drop both first (two 14B caches), then allocate"""
    for st in ((HI, LO) if a.prog else (HI,)):
        st["_caches"] = None
    for st in ((HI, LO) if a.prog else (HI,)):
        use(st); b.reset_caches(); st.update(snapshot())
# ---- the progressive student step for one chunk (shared by the no-grad rollout and the with-grad replay) ----
def student_chunk(c, g, stop, zin_store=None, zin_given=None):
    """stop in {1,2}: returns x (sample of this chunk at the full grid). zin_store (list) records the latent entering the stop step; zin_given replays it."""
    x_full = None
    for j in range(stop):
        s_ = SIG[j]
        if a.prog and j == 0:
            use(LO); adapter("gen", step_lo=True); e0 = torch.randn(HI["_chunk_shape"], generator=g).to(dev) if zin_given is None else None
            z = pool_noise(e0.float(), PH, PW) if zin_given is None else zin_given
            if j == stop - 1 and zin_store is not None: zin_store.append(z.detach().clone())
            x_lo = b.target_x0(z, j, c).float(); x_full = up(x_lo, HW)
        else:
            use(HI); adapter("gen", step_lo=False)
            if j == 0: z = b.chunk_noise(c, g) if zin_given is None else zin_given
            else: z = (1 - s_) * x_full.detach() + s_ * b.randn_like(x_full, g) if zin_given is None else zin_given
            if j == stop - 1 and zin_store is not None: zin_store.append(z.detach().clone())
            x_full = b.target_x0(z, j, c).float()
    return x_full
def commit_both(x, c):
    with torch.no_grad():
        use(HI); adapter("gen", step_lo=False); b.commit(x, c)
        if a.prog: use(LO); adapter("gen", step_lo=True); b.commit(down(x, LW), c)
@torch.no_grad()
def rollout(Nc, seed, stops):
    reset_states(); g = gen(seed, 0); xs = []; zin = []
    for c in range(Nc):
        x = student_chunk(c, g, stops[c], zin_store=zin); commit_both(x, c); xs.append(x)
    return torch.cat(xs, 1), zin
def fake_x0_clip(z, x, sig, grad=False, loss_tgt=None):
    """critic x0 for the noisy FULL-RES clip z, teacher-forced on the clean clip x (its own causal context in the HI state)"""
    use(HI); adapter("fake"); b.reset_caches(); HI.update(snapshot()); Nc = x.shape[1] // CS; outs = []; losses = []
    for c in range(Nc):
        zc = z[:, CS * c:CS * c + CS]
        with torch.set_grad_enabled(grad):
            u = b.target_flow(zc, float(sig), c)
            if grad: l = Fn.mse_loss(u, loss_tgt[:, CS * c:CS * c + CS]); (l / Nc).backward(); losses.append(float(l))
            else: outs.append((zc - float(sig) * u).detach())
        with torch.no_grad(): b.commit(x[:, CS * c:CS * c + CS], c)
    return (torch.cat(outs, 1) if outs else None), losses
def pick(): return random.choice(TASKS), random.choice(STREAMS), random.randrange(1 << 30), random.choice(AREAS)
# ---- real score: local bid on cuda:1 or the remote server ----
if not a.dist:
    b1 = build_bid(dev1)
    def get_real(task, area, pl, z, sig): return real_x0(b1, z, sig, bid_cond(b1, task, area, pl)).to(dev)
else:
    def get_real(task, area, pl, z, sig):
        F_, h, w = z.shape[1:]; hdr = torch.tensor([TASKS.index(task), area, F_, h, w, 0, int(sig * 1e6), pl.shape[1]], dtype=torch.long); dst = WORLD - 1
        dist.send(hdr, dst=dst); dist.send(z.float().cpu().contiguous(), dst=dst); dist.send(pl.float().cpu().contiguous(), dst=dst)
        x0 = torch.zeros_like(z, device="cpu"); dist.recv(x0, src=dst); return x0.to(dev)
def allreduce_grads(params):
    if a.dist and not a.fsdp and len(GEN_RANKS) > 1:
        for p in params:
            if p.grad is not None: dist.all_reduce(p.grad, group=GEN_GROUP); p.grad /= len(GEN_RANKS)
H = dict(dmd=[], gn=[], fake=[], sig=[]); t0 = time.time(); TM = dict(roll=0.0, bid=0.0, fake=0.0, replay=0.0, critic=0.0)
logw("start step %d: prog %d nodes %s sigmas %s adapt %s (gen params %.0fM, fake LoRA r%d) lr gen %g fake %g critic ratio %d grad chunks %d frames %d areas %s window %d dist %d world %d" % (
    step, a.prog, NODES, [round(s, 3) for s in SIG], a.adapt, sum(p.numel() for p in P_GEN) / 1e6, a.rank_fake, a.lr_gen, a.lr_fake, a.critic_ratio, a.grad_chunks, a.frames, AREAS, a.window, a.dist, WORLD))
if a.smoke: a.steps = step + a.smoke
while step < a.steps:
    # ---- critic steps: flow matching of the fake score on fresh generator rollouts ----
    for _ in range(a.critic_ratio):
        _t = time.time(); task, stream, seed, area = pick(); Nc, pl = setup_task(task, stream, area); stops = [sample_stop() for _ in range(Nc)]
        x, _ = rollout(Nc, seed, stops); sig = sample_sig(); eps = torch.randn_like(x); z = (1 - sig) * x + sig * eps
        _, ls = fake_x0_clip(z, x, sig, grad=True, loss_tgt=eps - x); allreduce_grads(P_FAKE)
        torch.nn.utils.clip_grad_norm_(P_FAKE, 1.0); opt_f.step(); opt_f.zero_grad(set_to_none=True); H["fake"].append(float(np.mean(ls))); TM["critic"] += time.time() - _t
    # ---- generator step ----
    _t = time.time(); task, stream, seed, area = pick(); Nc, pl = setup_task(task, stream, area); stops = [sample_stop() for _ in range(Nc)]
    x, zin = rollout(Nc, seed, stops); torch.cuda.synchronize(); TM["roll"] += time.time() - _t
    _t = time.time(); sig = sample_sig(); eps = torch.randn_like(x); z = (1 - sig) * x + sig * eps
    x0_real = get_real(task, area, pl, z, sig); torch.cuda.synchronize(); TM["bid"] += time.time() - _t
    _t = time.time(); x0_fake, _ = fake_x0_clip(z, x, sig); torch.cuda.synchronize(); TM["fake"] += time.time() - _t
    with torch.no_grad():
        nrm = (x - x0_real).abs().mean().clamp_min(1e-6); gvec = torch.nan_to_num((x0_fake - x0_real) / nrm); tgt = x - gvec
    H["gn"].append(float(gvec.norm() / x.norm())); H["sig"].append(sig)
    if a.sanity and step == 0 and RANK == 0 and not os.path.exists(a.out + "/sanity_gen.mp4"):
        with torch.no_grad():
            use(HI)
            for nm_, lat in (("gen", x), ("real", x0_real), ("fake", x0_fake)):
                os.makedirs(a.out + "/sanity_" + nm_, exist_ok=True); b.decode_and_save([lat[:, CS * c:CS * c + CS] for c in range(Nc)], a.out + "/sanity_" + nm_)
                if os.path.exists(a.out + "/sanity_%s/spec.mp4" % nm_): os.replace(a.out + "/sanity_%s/spec.mp4" % nm_, a.out + "/sanity_%s.mp4" % nm_)
        logw("sanity decodes written (task %s stream %s area %d sigma %.2f stops %s)" % (task, stream, area, sig, stops)); torch.cuda.empty_cache()
    # ---- replay the stop step of `grad-chunks` chunks WITH grad (history re-committed from x without grad); k = 1 -> grad flows through up() into the half-res step ----
    _t = time.time(); reset_states(); G = set(random.sample(range(Nc), min(a.grad_chunks, Nc))); dl = []
    for c in range(Nc):
        if c in G:
            xc = student_chunk(c, None, stops[c], zin_given=zin[c]) if stops[c] == 1 else None
            if stops[c] == 2:   # replay only the full-res step from the recorded re-noised latent (the half-res step is not on the gradient path for k = 2)
                use(HI); adapter("gen", step_lo=False); xc = b.target_x0(zin[c], 1, c).float()
            l = 0.5 * Fn.mse_loss(xc, tgt[:, CS * c:CS * c + CS]); (l / len(G)).backward(); dl.append(float(l))
        commit_both(x[:, CS * c:CS * c + CS], c)
    H["dmd"].append(float(np.mean(dl))); torch.cuda.synchronize(); TM["replay"] += time.time() - _t
    allreduce_grads(P_GEN); torch.nn.utils.clip_grad_norm_(P_GEN, 1.0); opt_g.step(); opt_g.zero_grad(set_to_none=True); step += 1
    if EMA is not None:
        with torch.no_grad():
            for e_, p in zip(EMA, P_GEN): e_.mul_(a.ema).add_(p.detach(), alpha=1 - a.ema)
    if step % a.log_every == 0 or a.smoke:
        logw("step %d  area %d sig %.2f stops %s  dmd %.4f |grad|/|x| %.3f  fake %.4f  %.0f min  %.1f GiB  t(roll/bid/fake/replay/critic) %.0f/%.0f/%.0f/%.0f/%.0f s" % (
            step, area, sig, "".join(str(s_) for s_ in stops), np.mean(H["dmd"][-a.log_every:]), np.mean(H["gn"][-a.log_every:]), np.mean(H["fake"][-a.log_every * a.critic_ratio:] or [0]), (time.time() - t0) / 60,
            torch.cuda.max_memory_allocated(dev) / 2**30, TM["roll"], TM["bid"], TM["fake"], TM["replay"], TM["critic"]))
    if (step % a.save_every == 0 or step >= a.steps) and RANK == 0:
        save_student(step, "_s%d" % step); save_student(step, "_last"); save_fake(step); json.dump(dict(step=step, args=vars(a), H={k_: v[-2000:] for k_, v in H.items()}), open(a.out + "/state.json", "w"))
        if EMA is not None:
            bak = [p.detach().clone() for p in P_GEN]
            with torch.no_grad():
                for e_, p in zip(EMA, P_GEN): p.copy_(e_)
            save_student(step, "_ema_last")
            with torch.no_grad():
                for b_, p in zip(bak, P_GEN): p.copy_(b_)
if a.dist:
    hdr = torch.tensor([0, 0, 0, 0, 0, 1, 0, 0], dtype=torch.long); dist.send(hdr, dst=WORLD - 1); dist.barrier()
print("[PROG_DMD14_DONE] %d steps, %.0f min" % (step, (time.time() - t0) / 60), flush=True)
