"""[10/13, lever 1] Self-distillation of the 14B's FIRST step: a LoRA that is active ONLY at node 0 (t = 1000) and is trained so that the
step-1 x0 estimate matches the 14B's own later output (--target k: x0 after node k of the SAME 4-step run, default 3 = the final chunk).
Context = the 14B's own committed latents (teacher forcing, GATE off during commit, as in deployment where the committed history is the
final output). Nodes 1-3 and the commit pass never see the LoRA, so the texture steps are untouched (facts §22: a node-shared LoRA blurs).
Data: specwm/target_dump.py clips (lat.pt, x0.pt {node: [N,16,3,h,w]}). Loss = ||x0_1 - x0_k||^2 / ||x0_k||^2 on 1x2x2 tokens; the
validation also reports the share of tokens within eps = 0.24 sigma_T of the FINAL output (the acceptance measure of facts §21).
Saves <out>/lora_s<step>.pt (A/B lists in block order q,k,v,o[,ffn0,ffn2]; gate=node0) and <out>/lora.pt (best val loss).
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.self_distill14 --data specwm_out/distill_data47 --out specwm_out/selfd14/T3 --target 3"""
import argparse, os, sys, glob, json, time, math, random, numpy as np, torch, torch.nn as nn
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--rank", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--ffn", type=int, default=1); ap.add_argument("--val", type=int, default=12); ap.add_argument("--val-every", type=int, default=250)
ap.add_argument("--target", type=int, default=3, help="node whose x0 is the regression target for the step-1 estimate (3 = final output)")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-steps", type=int, default=1500); ap.add_argument("--eps", type=float, default=0.24)
ap.add_argument("--val-clips", default="", help="comma list of clip dirs (basenames) to hold out; default = first --val of a fixed shuffle")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); random.seed(a.seed); torch.manual_seed(a.seed); dev = torch.device("cuda:0")
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding
FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
b = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
_te = b.pipe.text_encoder
class _TE:
    def __init__(s, te): s.te = te; s.model = te.model
    def __call__(s, texts, device): return s.te(texts, dev)
b.pipe.text_encoder = _TE(_te)
m = b.target_model.eval(); sig = [float(s) for s in b.node_sigmas()]
for p in m.parameters(): p.requires_grad_(False)
GATE = torch.ones((), device=dev)   # 1 at node 0, 0 elsewhere (commit, nodes 1-3)
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
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
TT = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
clips = sorted(d for d in glob.glob(a.data + "/*_s*") if os.path.exists(d + "/x0.pt"))
if a.val_clips: vs = set(a.val_clips.split(",")); VAL = [d for d in clips if os.path.basename(d) in vs]; TRAIN = [d for d in clips if os.path.basename(d) not in vs]
else: random.Random(1).shuffle(clips); VAL, TRAIN = clips[:a.val], clips[a.val:]
print("[selfd] clips %d train / %d val, LoRA %.1fM params, target node %d, sigmas %s" % (len(TRAIN), len(VAL), sum(p.numel() for p in params) / 1e6, a.target, sig), flush=True)
def gen(seed, c, path): return torch.Generator().manual_seed(seed * 1009 + c * 7 + {"node0": 0, "D": 1, "T": 2}[path])
def pred0(z, ci, grad):   # step-1 x0 estimate with the LoRA on
    GATE.fill_(1.0)
    with torch.set_grad_enabled(grad), torch.amp.autocast("cuda", dtype=b.param_dtype):
        out = m(x=[z.to(b.param_dtype)], t=torch.full((1,), float(b._timesteps[0]), device=dev), cross_attn_first_call=not b._cross_init["target"], **b._kwargs("target", ci))[0]
    b._cross_init["target"] = True; return z - sig[0] * out.float()   # GATE stays 1 until after backward (checkpoint recomputes the forward in backward); commit paths reset it
def err(xh, x0): return ((xh - x0) ** 2).mean() / (x0 ** 2).mean().clamp_min(1e-6)
def patch(x):
    c, f, h, w = x.shape; return x.view(c, f, 1, h // 2, 2, w // 2, 2).permute(1, 3, 5, 2, 4, 6, 0).reshape(f * (h // 2) * (w // 2), 64)
def within(xh, xF):
    t = patch(xF); sg = float(t.norm(dim=1).median()); return float(((patch(xh) - t).norm(dim=1) / sg <= a.eps).float().mean())
def clip_iter(d):
    meta = json.load(open(d + "/meta.json")); T = TT[meta["task"]]; seed = meta["seed"]; assert abs(meta["sigmas"][1] - sig[1]) < 1e-4, (meta["sigmas"], sig)
    L = torch.load(d + "/lat.pt").float().to(dev); X0 = torch.load(d + "/x0.pt"); N = meta["chunks"]
    _te.model.to(dev)   # T5 (11 GB) lives on the GPU only while encoding: peak was 91.6 GiB with it resident
    with torch.no_grad(): b.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); b.reset_caches()
    _te.model.to("cpu"); torch.cuda.empty_cache()
    assert tuple(L.shape[2:]) == tuple(b._chunk_shape[2:]), (L.shape, b._chunk_shape)
    GATE.fill_(0.0)
    with torch.no_grad(): b.commit(L[:, :3], 0)
    for c in range(1, N):
        z = b.chunk_noise(c, gen(seed, c, "node0")); xk = X0[a.target][c].float().to(dev); xF = X0[3][c].float().to(dev); x1 = X0[0][c].float().to(dev)
        yield c, z, xk, xF, x1
        GATE.fill_(0.0)
        with torch.no_grad(): b.commit(L[:, 3 * c:3 * c + 3], c)
@torch.no_grad()
def validate():
    e, wF, w1, wS = [], [], [], []
    for d in VAL:
        for c, z, xk, xF, x1 in clip_iter(d):
            xh = pred0(z, c, False); e.append(float(err(xh, xk))); wF.append(within(xh, xF)); wS.append(within(x1, xF)); w1.append(within(xh, x1))
    return dict(loss=float(np.mean(e)), withinF=float(np.mean(wF)), withinF_stock=float(np.mean(wS)), within_stock1=float(np.mean(w1)))
log = open(os.path.join(a.out, "log.txt"), "a"); step = 0; t0 = time.time(); hist = []
def fmt(v): return "loss %.4f  within-eps of final %.1f%% (stock step-1 %.1f%%, dumped stock step-1 vs ours %.1f%%)" % (v["loss"], 100 * v["withinF"], 100 * v["withinF_stock"], 100 * v["within_stock1"])
v = validate(); msg = "step 0 val " + fmt(v); print("[selfd]", msg, flush=True); log.write(msg + "\n"); log.flush(); best = v["loss"]
def save(tag): torch.save(dict(rank=a.rank, ffn=a.ffn, step=step, gate="node0", target=a.target, A=[l.A.detach().cpu() for l in LOR], B=[l.B.detach().cpu() for l in LOR]), os.path.join(a.out, "lora%s.pt" % tag))
VAL_DUE = [False]
for ep in range(a.epochs):
    random.shuffle(TRAIN)
    for d in TRAIN:
        for c, z, xk, xF, x1 in clip_iter(d):
            loss = err(pred0(z, c, True), xk); opt.zero_grad(set_to_none=True); loss.backward(); GATE.fill_(0.0); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); step += 1; hist.append(float(loss))
            if step % 50 == 0:
                msg = "step %d ep %d train loss %.4f  %.0f min  %.1f GiB  |B| %.2e" % (step, ep, np.mean(hist[-50:]), (time.time() - t0) / 60, torch.cuda.max_memory_allocated() / 2**30, float(LOR[0].B.norm())); print("[selfd]", msg, flush=True); log.write(msg + "\n"); log.flush()
            if step % a.val_every == 0: VAL_DUE[0] = True
            if a.max_steps and step >= a.max_steps: break
        if VAL_DUE[0]:
            VAL_DUE[0] = False; v = validate(); msg = "step %d val " % step + fmt(v); print("[selfd]", msg, flush=True); log.write(msg + "\n"); log.flush(); save("_s%d" % step)
            if v["loss"] < best: best = v["loss"]; save("")
        if a.max_steps and step >= a.max_steps: break
    if a.max_steps and step >= a.max_steps: break
v = validate(); msg = "final step %d val " % step + fmt(v); print("[selfd]", msg, flush=True); log.write(msg + "\n"); save("_last")
if v["loss"] < best: save("")
print("[SELFD_DONE] steps %d, %.0f min" % (step, (time.time() - t0) / 60), flush=True)
