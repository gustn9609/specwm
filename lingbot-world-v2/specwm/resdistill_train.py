"""Resolution distillation: teach the 14B at the low cloud resolution to behave like the 14B at 832x480. Data = 14B 832 2-step rollouts
(target_dump.py: lat.pt). Each clip's 832 video is decoded, resized to the low-res pixel grid of the cloud (exact grid from the binding at
--lo-area) and re-encoded once (cached as lo_<area>.pt). Training is teacher-forced at the low resolution: the 14B's K/V holds the
downscaled 832 chunks, and for every chunk c the LoRA learns the flow-matching x0-prediction of the downscaled 832 chunk at the two
sampler nodes (sigma 1.0 and 0.833) from fresh noise — i.e. what comes next in the 832 model's world, rendered at low resolution.
SDPA attention (Sage has no backward). Saves <out>/rd_lora.pt (same format as drift_rft_train: merge with res_split_infer --cloud-lora).
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.resdistill_train --data specwm_out/res_data --out specwm_out/resdistill/run1"""
import argparse, os, sys, glob, json, time, math, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--lo-area", type=int, default=175104); ap.add_argument("--rank", type=int, default=32); ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--ffn", type=int, default=0); ap.add_argument("--val", type=int, default=8); ap.add_argument("--val-every", type=int, default=400); ap.add_argument("--max-steps", type=int, default=0); ap.add_argument("--nodes", default="0,1", help="which sampler nodes get the loss; node 0 (sigma 1) is pure mean regression and collapsed textures in the first runs"); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); random.seed(0); torch.manual_seed(0); dev = torch.device("cuda:0")
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
b = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 500))
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="flash", ring_kv=True)
m = b.target_model.eval(); sig = [float(s) for s in b.node_sigmas()]
for p in m.parameters(): p.requires_grad_(False)
_te = b.pipe.text_encoder; _tc = {}
class _TE:
    def __init__(s): s.model = _te.model
    def __call__(s, texts, device):
        k = tuple(texts)
        if k not in _tc: _tc[k] = _te(texts, device)
        return _tc[k]
b.pipe.text_encoder = _TE()
class LoRA(nn.Module):
    def __init__(self, base, r):
        super().__init__(); self.base = base; self.A = nn.Parameter(torch.randn(r, base.in_features, device=dev) / math.sqrt(base.in_features)); self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
    def forward(self, x): return self.base(x) + ((x.float() @ self.A.t()) @ self.B.t()).to(x.dtype)
LOR = []
for blk in m.blocks:
    for nm in ("q", "k", "v", "o"): l = LoRA(getattr(blk.self_attn, nm), a.rank); setattr(blk.self_attn, nm, l); LOR.append(l)
    if a.ffn:
        for i in (0, 2): l = LoRA(blk.ffn[i], a.rank); blk.ffn[i] = l; LOR.append(l)
params = [p for l in LOR for p in (l.A, l.B)]
from torch.utils.checkpoint import checkpoint
for blk in m.blocks:
    _f = blk.forward
    def _ck(*args, _f=_f, **kw): return checkpoint(_f, *args, use_reentrant=False, **kw) if torch.is_grad_enabled() else _f(*args, **kw)
    blk.forward = _ck
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0); TT = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
clips = sorted(d for d in glob.glob(a.data + "/*_s*") if os.path.exists(d + "/meta.json") and os.path.exists(d + "/lat.pt")); random.Random(1).shuffle(clips); VAL, TRAIN = clips[:a.val], clips[a.val:]
print("[rd] clips %d train / %d val, LoRA %.1fM params, lo-area %d" % (len(TRAIN), len(VAL), sum(p.numel() for p in params) / 1e6, a.lo_area), flush=True)
@torch.no_grad()
def lo_latents(d, shape):   # decode the 832 rollout, resize to the cloud's low-res pixel grid, re-encode; cached per clip
    path = d + "/lo_%d.pt" % a.lo_area
    if os.path.exists(path): return torch.load(path).float().to(dev)
    L = torch.load(d + "/lat.pt").float().to(dev)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16): v = b.pipe.vae.decode([L])[0].float().clamp(-1, 1)
    H, W = 8 * shape[2], 8 * shape[3]; v = F.interpolate(v.permute(1, 0, 2, 3), size=(H, W), mode="bicubic", align_corners=False, antialias=True).clamp(-1, 1).permute(1, 0, 2, 3)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16): lo = b.pipe.vae.encode([v])[0].float()
    torch.save(lo.half().cpu(), path); return lo
def pred(z, k, ci, grad):
    with torch.set_grad_enabled(grad), torch.amp.autocast("cuda", dtype=b.param_dtype):
        out = m(x=[z.to(b.param_dtype)], t=torch.full((1,), float(b._timesteps[k]), device=dev), cross_attn_first_call=not b._cross_init["target"], **b._kwargs("target", ci))[0]
    b._cross_init["target"] = True; return z - sig[k] * out.float()
def err(xh, x0): return ((xh - x0) ** 2).mean() / (x0 ** 2).mean().clamp_min(1e-6)
def clip_iter(d):
    meta = json.load(open(d + "/meta.json")); T = TT[meta["task"]]; seed = meta["seed"]; N = meta["chunks"]
    b.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=a.lo_area); b.reset_caches(); shp = tuple(b._chunk_shape)
    lo = lo_latents(d, shp)
    if lo.shape[2:] != shp[2:]: return
    g = torch.Generator().manual_seed(seed * 7 + 3)
    with torch.no_grad(): b.commit(lo[:, :3], 0)
    for c in range(1, N):
        x = lo[:, 3 * c:3 * c + 3]; yield c, x, g
        with torch.no_grad(): b.commit(x, c)
def step_loss(c, x, g, grad):
    l = 0.0
    for k in [int(v) for v in a.nodes.split(",")]:
        z = (1 - sig[k]) * x + sig[k] * b.randn_like(x, g) if sig[k] < 0.999 else b.randn_like(x, g); l = l + err(pred(z, k, c, grad), x)
    return l
@torch.no_grad()
def validate(): return float(np.mean([float(step_loss(c, x, g, False)) for d in VAL for c, x, g in clip_iter(d)]))
log = open(os.path.join(a.out, "log.txt"), "a"); step = 0; t0 = time.time(); hist = []; v = validate(); msg = "step 0 val %.4f" % v; print("[rd]", msg, flush=True); log.write(msg + "\n"); best = v
def save(tag): torch.save(dict(rank=a.rank, ffn=a.ffn, step=step, lo_area=a.lo_area, A=[l.A.detach().cpu() for l in LOR], B=[l.B.detach().cpu() for l in LOR]), os.path.join(a.out, "rd_lora%s.pt" % tag))
for ep in range(a.epochs):
    random.shuffle(TRAIN)
    for d in TRAIN:
        for c, x, g in clip_iter(d):
            loss = step_loss(c, x, g, True); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); step += 1; hist.append(float(loss))
            if step % 50 == 0: msg = "step %d ep %d train %.4f  %.0f min" % (step, ep, np.mean(hist[-50:]), (time.time() - t0) / 60); print("[rd]", msg, flush=True); log.write(msg + "\n"); log.flush()
            if a.max_steps and step >= a.max_steps: break
        if step and (step // a.val_every) != ((step - 12) // a.val_every):
            v = validate(); msg = "step %d val %.4f" % (step, v); print("[rd]", msg, flush=True); log.write(msg + "\n"); save("_s%d" % step)
            if v < best: best = v; save("")
        if a.max_steps and step >= a.max_steps: break
    if a.max_steps and step >= a.max_steps: break
v = validate(); msg = "final step %d val %.4f" % (step, v); print("[rd]", msg, flush=True); log.write(msg + "\n"); save("_last")
if v < best: save("")
print("[RD_DONE] steps %d, %.0f min" % (step, (time.time() - t0) / 60), flush=True)
