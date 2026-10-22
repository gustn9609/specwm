"""Distil the 1.3B draft (LoRA) into the 14B target's 2-step sampler (data: specwm/target_dump.py).
For a dumped clip and chunk c >= 1, the draft's KV holds the 14B's own previous chunks (teacher forcing, committed at t = 0). With the dump's noise:
  node 0: z0 = chunk noise                         -> draft x0 must match the 14B's x0 at node 0
  node 1: z1 = (1 - s1) x0_T(node 0) + s1 eps1     -> draft x0 must match the 14B's x0 at node 1 (its final chunk)
loss = sum over nodes of ||x0_D - x0_T||^2 / ||x0_T||^2. The draft is then run with the 14B's node schedule (timesteps 0, 500).
Holdout loss on --val clips every --val-every steps. Saves <out>/draft_lora.pt (A/B lists in block order q,k,v,o[,ffn0,ffn2]) + log.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.draft_distill_train --data specwm_out/distill_data --out specwm_out/distill/run1"""
import argparse, os, sys, glob, json, time, math, random, numpy as np, torch, torch.nn as nn
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--rank", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--ffn", type=int, default=1); ap.add_argument("--val", type=int, default=12, help="number of clips held out for validation")
ap.add_argument("--val-every", type=int, default=400); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-steps", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); random.seed(a.seed); torch.manual_seed(a.seed); dev = torch.device("cuda:0")
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding
FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
b = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 500), cfg_name="i2v-1.3B", assets_dir=FAST)
m = b.target_model.eval(); sig = [float(s) for s in b.node_sigmas()]
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
TT = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
clips = sorted(d for d in glob.glob(a.data + "/*_s*") if os.path.exists(d + "/x0.pt"))
random.Random(1).shuffle(clips); VAL, TRAIN = clips[:a.val], clips[a.val:]
print("[distill] clips %d train / %d val, LoRA %.1fM params, sigmas %s" % (len(TRAIN), len(VAL), sum(p.numel() for p in params) / 1e6, sig), flush=True)
def gen(seed, c, path): return torch.Generator().manual_seed(seed * 1009 + c * 7 + {"node0": 0, "D": 1, "T": 2}[path])
def pred(z, k, ci, grad):
    with torch.set_grad_enabled(grad), torch.amp.autocast("cuda", dtype=b.param_dtype):
        out = m(x=[z.to(b.param_dtype)], t=torch.full((1,), float(b._timesteps[k]), device=dev), cross_attn_first_call=not b._cross_init["target"], **b._kwargs("target", ci))[0]
    b._cross_init["target"] = True; return z - sig[k] * out.float()
def err(xh, x0): return ((xh - x0) ** 2).mean() / (x0 ** 2).mean().clamp_min(1e-6)
SKIP = set()
def clip_iter(d, train):
    meta = json.load(open(d + "/meta.json")); T = TT[meta["task"]]; seed = meta["seed"]; assert abs(meta["sigmas"][1] - sig[1]) < 1e-4, (meta["sigmas"], sig)
    L = torch.load(d + "/lat.pt").float().to(dev); X0 = torch.load(d + "/x0.pt"); N = meta["chunks"]
    b.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); b.reset_caches()
    if tuple(L.shape[2:]) != tuple(b._chunk_shape[2:]) or X0[0].shape[0] < N:   # 14B and 1.3B grids differ for a few aspect ratios
        SKIP.add(d); return
    with torch.no_grad(): b.commit(L[:, :3], 0)
    for c in range(1, N):
        x0T0, x0T1 = X0[0][c].float().to(dev), X0[1][c].float().to(dev)
        z0 = b.chunk_noise(c, gen(seed, c, "node0")); eps1 = b.randn_like(x0T0, gen(seed, c, "T")); z1 = (1 - sig[1]) * x0T0 + sig[1] * eps1
        yield c, z0, z1, x0T0, x0T1
        with torch.no_grad(): b.commit(L[:, 3 * c:3 * c + 3], c)
@torch.no_grad()
def validate():
    e0, e1 = [], []
    for d in VAL:
        try:
            for c, z0, z1, x0T0, x0T1 in clip_iter(d, False): e0.append(float(err(pred(z0, 0, c, False), x0T0))); e1.append(float(err(pred(z1, 1, c, False), x0T1)))
        except RuntimeError as ex:
            if "Sizes of tensors must match" not in str(ex): raise
            SKIP.add(d)
    return float(np.mean(e0)), float(np.mean(e1))
log = open(os.path.join(a.out, "log.txt"), "a"); step = 0; t0 = time.time(); hist = []
v = validate(); msg = "step 0 val node0 %.4f node1 %.4f" % v; print("[distill]", msg, flush=True); log.write(msg + "\n"); log.flush(); best = sum(v)
def save(tag):
    torch.save(dict(rank=a.rank, ffn=a.ffn, step=step, A=[l.A.detach().cpu() for l in LOR], B=[l.B.detach().cpu() for l in LOR]), os.path.join(a.out, "draft_lora%s.pt" % tag))
VAL_DUE = [False]
for ep in range(a.epochs):
    random.shuffle(TRAIN)
    for d in TRAIN:
        try:
            for c, z0, z1, x0T0, x0T1 in clip_iter(d, True):
                l0 = err(pred(z0, 0, c, True), x0T0); l1 = err(pred(z1, 1, c, True), x0T1); loss = l0 + l1
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); step += 1; hist.append((float(l0), float(l1)))
                if step % 50 == 0:
                    h = np.mean(hist[-50:], 0); msg = "step %d ep %d train node0 %.4f node1 %.4f  %.0f min" % (step, ep, h[0], h[1], (time.time() - t0) / 60); print("[distill]", msg, flush=True); log.write(msg + "\n"); log.flush()
                if step % a.val_every == 0: VAL_DUE[0] = True
                if a.max_steps and step >= a.max_steps: break
        except RuntimeError as ex:   # a few aspect ratios give a 1.3B conditioning grid that differs from the dumped 14B latents
            if "Sizes of tensors must match" not in str(ex): raise
            SKIP.add(d); opt.zero_grad(set_to_none=True)
            print("[distill] skip clip", os.path.basename(d), "| y", tuple(b._y_chunks[0].shape), "chunk", tuple(b._chunk_shape), "| err", str(ex)[:100], flush=True)
        if VAL_DUE[0]:   # at a clip boundary: validate() re-prepares the shared binding for other tasks
            VAL_DUE[0] = False; v = validate(); msg = "step %d val node0 %.4f node1 %.4f" % (step, *v); print("[distill]", msg, flush=True); log.write(msg + "\n"); log.flush(); save("_s%d" % step)
            if sum(v) < best: best = sum(v); save("")
        if a.max_steps and step >= a.max_steps: break
    if a.max_steps and step >= a.max_steps: break
print("[distill] skipped clips (grid mismatch):", len(SKIP), sorted(os.path.basename(x) for x in SKIP)[:10], flush=True)
v = validate(); msg = "final step %d val node0 %.4f node1 %.4f" % (step, *v); print("[distill]", msg, flush=True); log.write(msg + "\n"); save("_last")
if sum(v) < best: save("")
print("[DISTILL_DONE] steps %d, %.0f min" % (step, (time.time() - t0) / 60), flush=True)
