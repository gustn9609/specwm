"""[10/14, item 2] Action-edit refiner: a LoRA on the 14B, active only in the 'edit' forward, trained so that ONE forward at node E (default 2,
sigma 0.833) turns a chunk generated under a wrong action (x_H, re-noised to node E) into the chunk the true action would have given (x_T,
same noise, same history). Data: specwm/edit_dump.py dirs (lat.pt = committed true chunks for history replay, xh.pt, meta.json rel_H / rel_T).
Context = the true history committed with the LoRA OFF. Loss = ||x_hat - x_T||^2 / ||x_T||^2 on the chunk; validation on held-out dirs
reports within-eps(x_hat, x_T) next to the as-is baseline within-eps(x_H, x_T) (facts §21/§27 measure). Saves <out>/lora_s<step>.pt and
<out>/lora.pt (best val loss).  usage: CUDA_VISIBLE_DEVICES=g python -m specwm.edit_train --data specwm_out/edit_data --val specwm_out/edit_data_val --out specwm_out/edit14/E2"""
import argparse, os, sys, glob, json, time, math, random, numpy as np, torch, torch.nn as nn
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--val", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--rank", type=int, default=64); ap.add_argument("--lr", type=float, default=3e-5); ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--node", type=int, default=2, help="noise node of the edit forward (0..3)")
ap.add_argument("--ffn", type=int, default=1); ap.add_argument("--val-every", type=int, default=250); ap.add_argument("--max-steps", type=int, default=3000); ap.add_argument("--eps", type=float, default=0.24); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); random.seed(a.seed); torch.manual_seed(a.seed); dev = torch.device("cuda:0")
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding
from wan.utils.cam_utils import get_Ks_transformed, get_plucker_embeddings
from einops import rearrange
FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
b = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
_te = b.pipe.text_encoder
class _TE:
    def __init__(s, te): s.te = te; s.model = te.model
    def __call__(s, texts, device): return s.te(texts, dev)
b.pipe.text_encoder = _TE(_te); m = b.target_model.eval(); sig = [float(s) for s in b.node_sigmas()]; CS = 3
for p in m.parameters(): p.requires_grad_(False)
GATE = torch.ones((), device=dev)
class LoRA(nn.Module):
    def __init__(self, base, r):
        super().__init__(); self.base = base; self.A = nn.Parameter(torch.randn(r, base.in_features, device=dev) / math.sqrt(base.in_features)); self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
    def forward(self, x): return self.base(x) + (GATE * ((x.float() @ self.A.t()) @ self.B.t())).to(x.dtype)
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
TRAIN = sorted(d for d in glob.glob(a.data + "/*_s*") if os.path.exists(d + "/meta.json")); VAL = sorted(d for d in glob.glob(a.val + "/*_s*") if os.path.exists(d + "/meta.json"))
print("[edit] clips %d train / %d val, LoRA %.1fM params, edit node %d (sigma %.3f)" % (len(TRAIN), len(VAL), sum(p.numel() for p in params) / 1e6, a.node, sig[a.node]), flush=True)
def patch(x):
    c, f, h, w = x.shape; return x.view(c, f, 1, h // 2, 2, w // 2, 2).permute(1, 3, 5, 2, 4, 6, 0).reshape(f * (h // 2) * (w // 2), 64)
def within(xh, xT):
    t = patch(xT); sg = float(t.norm(dim=1).median()); return float(((patch(xh) - t).norm(dim=1) / sg <= a.eps).float().mean())
def emb_chunk(rel, Ks, h, w, lat_h, lat_w):
    f = rel.shape[0]; e = get_plucker_embeddings(rel, Ks, h, w); e = rearrange(e, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)', c1=int(h // lat_h), c2=int(w // lat_w))[None, ...]
    return rearrange(e, 'b (f h w) c -> b c f h w', f=f, h=lat_h, w=lat_w).to(b.param_dtype)
def err(xh, x0): return ((xh - x0) ** 2).mean() / (x0 ** 2).mean().clamp_min(1e-6)
def edit_forward(xH, ci, g, grad):
    """re-noise x_H to node a.node with generator g and run ONE forward with the true-action embedding (LoRA on)"""
    s = sig[a.node]; z = (1 - s) * xH + s * torch.randn(xH.shape, generator=g).to(dev); GATE.fill_(1.0)
    with torch.set_grad_enabled(grad), torch.amp.autocast("cuda", dtype=b.param_dtype):
        out = m(x=[z.to(b.param_dtype)], t=torch.full((1,), float(b._timesteps[a.node]), device=dev), cross_attn_first_call=not b._cross_init["target"], **b._kwargs("target", ci))[0]
    b._cross_init["target"] = True; return z - s * out.float()   # GATE stays 1 until after backward (checkpoint recomputes the forward); commit paths reset it
def clip_iter(d):
    meta = json.load(open(d + "/meta.json")); T = TT[meta["task"]]; seed = meta["seed"]; stream = meta["stream"]
    L = torch.load(d + "/lat.pt").float().to(dev); XH = torch.load(d + "/xh.pt"); N = L.shape[1] // CS
    _te.model.to(dev)
    with torch.no_grad(): b.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); b.reset_caches()
    _te.model.to("cpu"); torch.cuda.empty_cache(); lat_h, lat_w = b._chunk_shape[2], b._chunk_shape[3]; h, w = lat_h * 8, lat_w * 8
    rel = torch.from_numpy(np.load(S + "/data/actstream/%s/rel.npy" % stream)).float().to(dev)[:N * CS]
    Ks = get_Ks_transformed(torch.from_numpy(np.load(S + "/data/actstream/%s/intrinsics.npy" % stream)).float(), height_org=480, width_org=832, height_resize=h, width_resize=w, height_final=h, width_final=w)[0].repeat(N * CS, 1).to(dev)
    b._emb_chunks = [emb_chunk(rel[CS * c:CS * c + CS], Ks[CS * c:CS * c + CS], h, w, lat_h, lat_w) for c in range(N)]   # true actions
    GATE.fill_(0.0)
    with torch.no_grad(): b.commit(L[:, :CS], 0)
    for c in range(1, N):
        row = meta["chunks"][c]
        if XH[c] is not None: yield c, XH[c].float().to(dev), L[:, CS * c:CS * c + CS], row, seed
        GATE.fill_(0.0)
        with torch.no_grad(): b.commit(L[:, CS * c:CS * c + CS], c)
@torch.no_grad()
def validate():
    e, wE, wH, by = [], [], [], {}
    for d in VAL:
        for c, xH, xT, row, seed in clip_iter(d):
            xh = edit_forward(xH, c, torch.Generator().manual_seed(seed * 1009 + c * 7 + 5), False); GATE.fill_(0.0); e.append(float(err(xh, xT))); wE.append(within(xh, xT)); wH.append(within(xH, xT)); by.setdefault(row["kind"], []).append((within(xh, xT), within(xH, xT)))
    return dict(loss=float(np.mean(e)), within_edit=float(np.mean(wE)), within_asis=float(np.mean(wH)), by_kind={k: (float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v])), len(v)) for k, v in by.items()})
def fmt(v): return "loss %.4f  within-eps edit %.1f%% (as-is %.1f%%)  %s" % (v["loss"], 100 * v["within_edit"], 100 * v["within_asis"], " ".join("%s %.0f/%.0f%%(n%d)" % (k, 100 * x[0], 100 * x[1], x[2]) for k, x in v["by_kind"].items()))
log = open(os.path.join(a.out, "log.txt"), "a"); step = 0; t0 = time.time(); hist = []
v = validate(); msg = "step 0 val " + fmt(v); print("[edit]", msg, flush=True); log.write(msg + "\n"); log.flush(); best = v["loss"]
def save(tag): torch.save(dict(rank=a.rank, ffn=a.ffn, step=step, node=a.node, A=[l.A.detach().cpu() for l in LOR], B=[l.B.detach().cpu() for l in LOR]), os.path.join(a.out, "lora%s.pt" % tag))
VAL_DUE = [False]
for ep in range(a.epochs):
    random.shuffle(TRAIN)
    for d in TRAIN:
        for c, xH, xT, row, seed in clip_iter(d):
            loss = err(edit_forward(xH, c, torch.Generator().manual_seed(random.randrange(1 << 30)), True), xT); opt.zero_grad(set_to_none=True); loss.backward(); GATE.fill_(0.0); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); step += 1; hist.append(float(loss))
            if step % 50 == 0:
                msg = "step %d ep %d train loss %.4f  %.0f min  %.1f GiB" % (step, ep, np.mean(hist[-50:]), (time.time() - t0) / 60, torch.cuda.max_memory_allocated() / 2**30); print("[edit]", msg, flush=True); log.write(msg + "\n"); log.flush()
            if step % a.val_every == 0: VAL_DUE[0] = True
            if a.max_steps and step >= a.max_steps: break
        if VAL_DUE[0]:
            VAL_DUE[0] = False; v = validate(); msg = "step %d val " % step + fmt(v); print("[edit]", msg, flush=True); log.write(msg + "\n"); log.flush(); save("_s%d" % step)
            if v["loss"] < best: best = v["loss"]; save("")
        if a.max_steps and step >= a.max_steps: break
    if a.max_steps and step >= a.max_steps: break
v = validate(); msg = "final step %d val " % step + fmt(v); print("[edit]", msg, flush=True); log.write(msg + "\n"); save("_last")
if v["loss"] < best: save("")
print("[EDIT_DONE] steps %d, %.0f min" % (step, (time.time() - t0) / 60), flush=True)
