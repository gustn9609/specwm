"""Mixed-memory LoRA: teach one Base expert to read an old prefix given as K x merged low-res tokens.

Teacher = the frozen expert on the full-res RePaint window (sink + whole prefix + new chunk, all noised at s).
Student = the same expert through base_mixed-style mixed forward: sink and last R prefix latents + new chunk at full res,
the old part [3, L-R) as K x downsampled (merged) tokens; LoRA deltas are applied ONLY to the merged tokens
(full-res tokens run exactly Base), so the adapter only changes what the old part contributes through attention.
Loss = MSE between student and teacher flow on the new chunk (relative to teacher energy). CFG: cond / uncond context at random.
Data: latents of Base 70-step rollouts of the training scenes (mixlora_cache.py); the test tasks (jar, castle) are held out.
Launch: torchrun --nproc_per_node 2 -m specwm.mixlora_train --expert high --out DIR   (CUDA_VISIBLE_DEVICES = the two GPUs)"""
import argparse, os, sys, glob, json, time, math, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser()
ap.add_argument("--expert", required=True, choices=["high", "low"]); ap.add_argument("--out", required=True)
ap.add_argument("--cache", default=S + "/lingbot-world-v2/specwm_out/mixlora/cache"); ap.add_argument("--K", type=int, default=4); ap.add_argument("--R", type=int, default=6)
ap.add_argument("--rank", type=int, default=32); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--cmin", type=int, default=4); ap.add_argument("--cmax", type=int, default=12); ap.add_argument("--save-every", type=int, default=100)
ap.add_argument("--ffn", type=int, default=1, help="also LoRA on the FFN linears"); ap.add_argument("--time-it", type=int, default=0)
a = ap.parse_args()
dist.init_process_group("nccl"); rank = dist.get_rank(); world = dist.get_world_size(); torch.cuda.set_device(rank); dev = torch.device("cuda", rank)
os.makedirs(a.out, exist_ok=True); random.seed(1000 + rank); torch.manual_seed(1000 + rank)
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world"); os.chdir(S + "/lingbot-world")
import importlib, importlib.util
spec = importlib.util.spec_from_file_location("specwm_v1", S + "/lingbot-world/specwm/__init__.py", submodule_search_locations=[S + "/lingbot-world/specwm"]); pkg = importlib.util.module_from_spec(spec); sys.modules["specwm_v1"] = pkg; spec.loader.exec_module(pkg)
V1Binding = importlib.import_module("specwm_v1.binding_v1").V1Binding
b1 = V1Binding(S + "/lingbot-world/lingbot-world-base-cam", t5_cpu=True, device=str(dev))
m = (b1.pipe.high_noise_model if a.expert == "high" else b1.pipe.low_noise_model).to(dev).eval(); b1.pipe.high_noise_model = b1.pipe.low_noise_model = None
torch.cuda.empty_cache()   # VAE kept: prepare() encodes the conditioning image
for p in m.parameters(): p.requires_grad_(False)
sys.path.insert(0, S + "/lingbot-world-v2")
from specwm.base_mixed import _rope_freqs, _rope, down, down_cam
from wan.modules.model import sinusoidal_embedding_1d
from einops import rearrange

# ---------------- masked LoRA (active only on merged-token positions) ----------------
MASK = [None]   # [1, L, 1] float mask (1 = merged old token) or None (LoRA off)
class LoRA(nn.Module):
    def __init__(self, base, r):
        super().__init__(); self.base = base; self.A = nn.Parameter(torch.randn(r, base.in_features, device=dev) / math.sqrt(base.in_features)); self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
    def forward(self, x):
        y = self.base(x)
        if MASK[0] is None: return y
        return y + ((x.float() @ self.A.t()) @ self.B.t()).to(y.dtype) * MASK[0].to(y.dtype)
LOR = []
for blk in m.blocks:
    sa = blk.self_attn
    for nm in ("q", "k", "v", "o"):
        l = LoRA(getattr(sa, nm), a.rank); setattr(sa, nm, l); LOR.append(l)
    if a.ffn:
        for i in (0, 2):
            l = LoRA(blk.ffn[i], a.rank); blk.ffn[i] = l; LOR.append(l)
params = [p for l in LOR for p in (l.A, l.B)]
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
for p in params: dist.broadcast(p.data, 0)
if rank == 0: print("[mixlora] expert %s  LoRA params %.1fM  world %d" % (a.expert, sum(p.numel() for p in params) / 1e6, world), flush=True)

# ---------------- data ----------------
SC = {sc: dict(img=S + "/lingbot-world-v2/examples/%s/image.jpg" % sc, prompt=open(S + "/p_%s.txt" % sc).read().strip(), act=S + "/rt2_action_%s" % sc) for sc in ("00", "01", "02", "05")}
FILES = sorted(glob.glob(a.cache + "/*.pt")); assert FILES, "no cache"
CONDC = {}
def cond(sc, F):
    if (sc, F) not in CONDC: CONDC[(sc, F)] = b1.prepare(SC[sc]["img"], SC[sc]["prompt"], SC[sc]["act"], frame_num=F, max_area=480 * 832)
    return CONDC[(sc, F)]
TS = [999, 993, 985, 977, 967, 956] if a.expert == "high" else [943, 927, 908, 885, 856, 819, 768, 697, 587, 399]   # the 16-step shift-10 schedule

# ---------------- student: mixed forward with grad (checkpointed blocks) ----------------
def block(blk, h, e0, FI, dcd, ctx, L):
    with torch.amp.autocast("cuda", dtype=torch.float32):
        ee = (blk.modulation.unsqueeze(0) + e0).chunk(6, dim=2)
    sa = blk.self_attn; n, d = sa.num_heads, sa.head_dim; dt = torch.bfloat16
    xin = (blk.norm1(h).float() * (1 + ee[1].squeeze(2)) + ee[0].squeeze(2)).to(dt)
    q = sa.norm_q(sa.q(xin)).view(1, L, n, d); k_ = sa.norm_k(sa.k(xin)).view(1, L, n, d); v = sa.v(xin).view(1, L, n, d)
    q = _rope(q, FI).to(dt); k_ = _rope(k_, FI).to(dt)
    o = Fn.scaled_dot_product_attention(q.transpose(1, 2), k_.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2).reshape(1, L, n * d)
    with torch.amp.autocast("cuda", dtype=torch.float32):
        h = h + sa.o(o) * ee[2].squeeze(2)
    if dcd is not None:
        ch = blk.cam_injector_layer2(Fn.silu(blk.cam_injector_layer1(dcd))) + dcd
        h = (1.0 + blk.cam_scale_layer(ch)) * h + blk.cam_shift_layer(ch)
    h = h + blk.cross_attn(blk.norm3(h), ctx, None)
    yf = blk.ffn((blk.norm2(h).float() * (1 + ee[4].squeeze(2)) + ee[3].squeeze(2)))
    with torch.amp.autocast("cuda", dtype=torch.float32):
        h = h + yf * ee[5].squeeze(2)
    return h
def student(segs, t_val, context, merged_idx):
    """returns flow of the last segment. merged_idx: index of the merged segment (LoRA mask)."""
    dt = torch.bfloat16
    with torch.amp.autocast("cuda", dtype=dt):
        toks, grids, cams = [], [], []
        for s in segs:
            u = m.patch_embedding(torch.cat([s["x"], s["y"]], 0).unsqueeze(0).to(dev, dt)); grids.append(tuple(u.shape[2:])); toks.append(u.flatten(2).transpose(1, 2))
            cams.append(rearrange(s["cam"].to(dev), '1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)', c1=m.patch_size[0], c2=m.patch_size[1], c3=m.patch_size[2]))
        h = torch.cat(toks, 1); Ls = [t.size(1) for t in toks]; L = h.size(1)
        msk = torch.zeros(1, L, 1, device=dev); o0 = sum(Ls[:merged_idx]); msk[:, o0:o0 + Ls[merged_idx]] = 1; MASK[0] = msk
        with torch.amp.autocast("cuda", dtype=torch.float32):
            t = torch.full((1, L), float(t_val), device=dev)
            e = m.time_embedding(sinusoidal_embedding_1d(m.freq_dim, t.flatten()).unflatten(0, (1, L)).float()); e0 = m.time_projection(e).unflatten(2, (6, m.dim))
        ctx = m.text_embedding(torch.stack([torch.cat([cc, cc.new_zeros(m.text_len - cc.size(0), cc.size(1))]) for cc in context]))
        FI = torch.cat([_rope_freqs(g, m.freqs, s["f0"], s["k"], m.blocks[0].self_attn.head_dim // 2) for s, g in zip(segs, grids)], 0)
        pe = m.patch_embedding_wancamctrl(torch.cat(cams, 1).to(dt)); dcd = pe + m.c2ws_hidden_states_layer2(Fn.silu(m.c2ws_hidden_states_layer1(pe)))
        for blk in m.blocks:
            h = checkpoint(block, blk, h, e0, FI, dcd, ctx, L, use_reentrant=False)
        out = m.head(h, e)
    # MASK stays set until after backward (checkpoint recomputation needs it)
    o0 = sum(Ls[:-1]); return m.unpatchify(out[:, o0:o0 + Ls[-1]], torch.tensor([grids[-1]]))[0].float()

# ---------------- training loop ----------------
if m.freqs.device != dev: m.freqs = m.freqs.to(dev)
log = open(os.path.join(a.out, "log_rank%d.txt" % rank), "a"); t_start = time.time(); ema = None
for step in range(1, a.steps + 1):
    t0 = time.time()
    f = random.choice(FILES); D = torch.load(f); lat = D["lat"].to(dev).float(); sc = D["scene"]
    ci = random.randint(a.cmin, a.cmax); L = 3 * ci; hi = max(4, L - a.R); X0 = lat[:, :L + 3]; F = 4 * (L + 3 - 1) + 1
    c = cond(sc, F); cam = c["dit_cond"]["c2ws_plucker_emb"][0]; y = c["y"]; tv = float(random.choice(TS)) + random.uniform(-3, 3); s = tv / 1000.0
    ctxs = [c["ctx"][0]] if random.random() < 0.5 else c["ctxn"]
    eps = torch.randn_like(X0); Xt = (1 - s) * X0 + s * eps
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        tgt = m([Xt.to(torch.bfloat16)], t=torch.full((1,), tv, device=dev), context=ctxs, seq_len=c["seq_len"], y=[y], dit_cond_dict=c["dit_cond"])[0].float()[:, L:]
    t_teacher = time.time() - t0
    old = (1 - s) * down(X0[:, 3:hi], a.K) + s * a.K * down(eps[:, 3:hi], a.K)
    segs = [dict(x=Xt[:, :3], y=y[:, :3], f0=0, k=1, cam=cam[:, :, :3]), dict(x=old, y=down(y[:, 3:hi], a.K), f0=3, k=a.K, cam=down_cam(cam[:, :, 3:hi], a.K)),
            dict(x=Xt[:, hi:], y=y[:, hi:L + 3], f0=hi, k=1, cam=cam[:, :, hi:L + 3])]
    pred = student(segs, tv, ctxs, 1)[:, -3:]
    loss = ((pred - tgt) ** 2).mean() / (tgt ** 2).mean().clamp_min(1e-6)
    opt.zero_grad(set_to_none=True); loss.backward(); MASK[0] = None
    for p in params:
        if p.grad is not None: dist.all_reduce(p.grad); p.grad /= world
    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    lv = torch.tensor([loss.item()], device=dev); dist.all_reduce(lv); lv = float(lv) / world; ema = lv if ema is None else 0.95 * ema + 0.05 * lv
    msg = "step %d loss %.4f ema %.4f  ci %d t %d  %.1fs (teacher %.1fs)  elapsed %.0fm" % (step, lv, ema, ci, tv, time.time() - t0, t_teacher, (time.time() - t_start) / 60)
    log.write(msg + "\n"); log.flush()
    if rank == 0 and (step <= 5 or step % 10 == 0): print("[mixlora]", msg, flush=True)
    if rank == 0 and (step % a.save_every == 0 or step == a.steps):
        torch.save(dict(expert=a.expert, K=a.K, R=a.R, rank=a.rank, ffn=a.ffn, step=step, A=[l.A.detach().cpu() for l in LOR], B=[l.B.detach().cpu() for l in LOR]), os.path.join(a.out, "lora_%s.pt" % a.expert))
    if a.time_it and step >= a.time_it: break
dist.destroy_process_group()
