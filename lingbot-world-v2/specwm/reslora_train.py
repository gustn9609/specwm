"""Resolution-split refiner: a LoRA on the 1.3B causal-fast that turns the cloud's 576x320 latent (bilinearly resampled to the 832x480 grid, re-noised
to sigma_r) into the 14B's 832x480 chunk in ONE forward. Data: specwm/target_dump.py (lat.pt = 14B 832 chunks) + specwm/res_pairs.py (up.pt = the
downscaled/re-encoded/upsampled counterpart). Teacher-forced context: the 1.3B's KV holds the 14B's previous 832 chunks (committed at t = 0).
  z = (1 - s_r) up_c + s_r eps,  x0_D = z - s_r v_D(z, t_r)  ->  loss ||x0_D - lat_c||^2 / ||lat_c||^2 (+ --up-loss: baseline error of up_c itself, logged)
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.reslora_train --data specwm_out/res_data --out specwm_out/reslora/run1 [--ref-index 900]"""
import argparse, os, sys, glob, json, time, math, random, numpy as np, torch, torch.nn as nn
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--rank", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--ffn", type=int, default=1); ap.add_argument("--val", type=int, default=6); ap.add_argument("--val-every", type=int, default=200); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-steps", type=int, default=0)
ap.add_argument("--pix", type=float, default=0.0, help="weight of a pixel-space loss: decode one random latent frame of the prediction and the target with the VAE (no grad on the VAE) and add pix * (L1 + LPIPS-vgg)"); ap.add_argument("--in-noise", type=float, default=0.0, help="std of extra noise added to the upsampled input (robustness to real 576 rollouts)"); ap.add_argument("--ref-index", default="900", help="scheduler index(es) of the refine node(s), comma-separated: 900 -> sigma 0.357, 930 -> 0.273, 850 -> 0.468. With several, each training step picks one node at random (node 1 input = re-noised upsampled cloud latent; later nodes = re-noised target, as in standard diffusion training); inference runs them in order")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); random.seed(a.seed); torch.manual_seed(a.seed); dev = torch.device("cuda:0")
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding
FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
b = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=tuple([0] + [int(v) for v in a.ref_index.split(",")]), cfg_name="i2v-1.3B", assets_dir=FAST)
m = b.target_model.eval(); sig = [float(s) for s in b.node_sigmas()]; SR = sig[1]; NODES = list(range(1, len(sig)))
for p in m.parameters(): p.requires_grad_(False)
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
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
if a.pix > 0:
    import lpips; LP = lpips.LPIPS(net="vgg").to(dev).eval()
    for p_ in LP.parameters(): p_.requires_grad_(False)
    for p_ in b.pipe.vae.model.parameters(): p_.requires_grad_(False)
    def dec1(lat1):   # [16,1,h,w] latent -> [1,3,H,W] in [-1,1]; the VAE's first frame is an image
        with torch.amp.autocast("cuda", dtype=torch.bfloat16): return b.pipe.vae.decode([lat1])[0][:, 0][None].float().clamp(-1, 1)
    def pix_loss(xh, x0):
        j = random.randrange(xh.shape[1]); ph = dec1(xh[:, j:j + 1])
        with torch.no_grad(): p0 = dec1(x0[:, j:j + 1])
        return (ph - p0).abs().mean() + LP(ph, p0).mean()
TT = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
clips = sorted(d for d in glob.glob(a.data + "/*_s*") if os.path.exists(d + "/up.pt"))
random.Random(1).shuffle(clips); VAL, TRAIN = clips[:a.val], clips[a.val:]
print("[reslora] clips %d train / %d val, LoRA %.1fM params, refine sigmas %s" % (len(TRAIN), len(VAL), sum(p.numel() for p in params) / 1e6, [round(sig[k], 3) for k in NODES]), flush=True)
def gen(seed, c): return torch.Generator().manual_seed(seed * 7919 + c * 13 + 5)
def pred(z, ci, grad, k=1):
    with torch.set_grad_enabled(grad), torch.amp.autocast("cuda", dtype=b.param_dtype):
        out = m(x=[z.to(b.param_dtype)], t=torch.full((1,), float(b._timesteps[k]), device=dev), cross_attn_first_call=not b._cross_init["target"], **b._kwargs("target", ci))[0]
    b._cross_init["target"] = True; return z - sig[k] * out.float()
def err(xh, x0): return ((xh - x0) ** 2).mean() / (x0 ** 2).mean().clamp_min(1e-6)
SKIP = set()
def clip_iter(d):
    meta = json.load(open(d + "/meta.json")); T = TT[meta["task"]]; seed = meta["seed"]
    L = torch.load(d + "/lat.pt").float().to(dev); U = torch.load(d + "/up.pt").float().to(dev); N = meta["chunks"]
    b.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); b.reset_caches()
    if tuple(L.shape[2:]) != tuple(b._chunk_shape[2:]): SKIP.add(d); return
    with torch.no_grad(): b.commit(L[:, :3], 0)
    for c in range(1, N):
        x_hi = L[:, 3 * c:3 * c + 3]; u = U[:, 3 * c:3 * c + 3]; k = random.choice(NODES); base = u if k == 1 else x_hi
        if a.in_noise > 0 and k == 1: base = base + a.in_noise * torch.randn_like(base)
        z = (1 - sig[k]) * base + sig[k] * b.randn_like(u, gen(seed, c))
        yield c, z, u, x_hi, k
        with torch.no_grad(): b.commit(x_hi, c)
@torch.no_grad()
def validate():
    e, e_up = [], []
    for d in VAL:
        try:
            for c, z, u, x_hi, k in clip_iter(d): e.append(float(err(pred(z, c, False, k), x_hi))); e_up.append(float(err(u, x_hi)))
        except RuntimeError as ex:
            if "Sizes of tensors must match" not in str(ex): raise
            SKIP.add(d)
    return float(np.mean(e)), float(np.mean(e_up))
log = open(os.path.join(a.out, "log.txt"), "a"); step = 0; t0 = time.time(); hist = []
v = validate(); msg = "step 0 val refine %.4f (upsampled-input baseline %.4f)" % v; print("[reslora]", msg, flush=True); log.write(msg + "\n"); log.flush(); best = v[0]
def save(tag): torch.save(dict(rank=a.rank, ffn=a.ffn, step=step, ref_index=[int(v) for v in a.ref_index.split(",")], A=[l.A.detach().cpu() for l in LOR], B=[l.B.detach().cpu() for l in LOR]), os.path.join(a.out, "res_lora%s.pt" % tag))
VAL_DUE = [False]
for ep in range(a.epochs):
    random.shuffle(TRAIN)
    for d in TRAIN:
        try:
            for c, z, u, x_hi, k in clip_iter(d):
                xh = pred(z, c, True, k); loss = err(xh, x_hi) + (a.pix * pix_loss(xh, x_hi) if a.pix > 0 else 0.0); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); step += 1; hist.append(float(loss))
                if step % 50 == 0: msg = "step %d ep %d train %.4f  %.0f min" % (step, ep, np.mean(hist[-50:]), (time.time() - t0) / 60); print("[reslora]", msg, flush=True); log.write(msg + "\n"); log.flush()
                if step % a.val_every == 0: VAL_DUE[0] = True
                if a.max_steps and step >= a.max_steps: break
        except RuntimeError as ex:
            if "Sizes of tensors must match" not in str(ex): raise
            SKIP.add(d); opt.zero_grad(set_to_none=True); print("[reslora] skip clip", os.path.basename(d), flush=True)
        if VAL_DUE[0]:
            VAL_DUE[0] = False; v = validate(); msg = "step %d val refine %.4f (baseline %.4f)" % (step, *v); print("[reslora]", msg, flush=True); log.write(msg + "\n"); log.flush(); save("_s%d" % step)
            if v[0] < best: best = v[0]; save("")
        if a.max_steps and step >= a.max_steps: break
    if a.max_steps and step >= a.max_steps: break
v = validate(); msg = "final step %d val refine %.4f (baseline %.4f)" % (step, *v); print("[reslora]", msg, flush=True); log.write(msg + "\n"); save("_last")
if v[0] < best: save("")
print("[RESLORA_DONE] steps %d, %.0f min, skipped %d" % (step, (time.time() - t0) / 60, len(SKIP)), flush=True)
