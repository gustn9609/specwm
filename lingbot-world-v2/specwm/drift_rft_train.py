"""Track A: drift-reduction fine-tune of the 14B at 576x320 by rejection-sampling self-distillation (data: drift_rft_dump.py). For each chunk the
model is teacher-forced on the kept (lowest-drift-score) trajectory and trained to reproduce the kept candidate's x0 at both nodes from that
candidate's noise — i.e. it learns to put more mass on the continuations the drift head accepts. LoRA on q,k,v,o(+ffn) of the FP8-swapped 14B.
Validation = held-out clips' relative error. Saves <out>/rft_lora.pt. usage: CUDA_VISIBLE_DEVICES=g python -m specwm.drift_rft_train --data specwm_out/rft_data --out specwm_out/rft/run1"""
import argparse, os, sys, glob, json, time, math, random, numpy as np, torch, torch.nn as nn
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--rank", type=int, default=32); ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--ffn", type=int, default=0); ap.add_argument("--val", type=int, default=8); ap.add_argument("--val-every", type=int, default=300); ap.add_argument("--max-steps", type=int, default=0); ap.add_argument("--dpo", type=float, default=0.0, help="beta > 0: Diffusion-DPO loss between the kept (winner) and worst (loser) candidate x0 targets, reference = the base model (LoRA switched off); needs x0_loser.pt"); ap.add_argument("--only-flagged", type=float, default=None, help="train only on chunks whose best-candidate logit improved by at least this over the worst (focus on chunks where selection mattered)"); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); random.seed(0); torch.manual_seed(0); dev = torch.device("cuda:0")
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
b = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 500))
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="flash", ring_kv=True)   # bf16 weights, SDPA attention: SageAttention has no backward (training with it diverged)
m = b.target_model.eval(); sig = [float(s) for s in b.node_sigmas()]
for p in m.parameters(): p.requires_grad_(False)
class LoRA(nn.Module):
    def __init__(self, base, r):
        super().__init__(); self.base = base; self.A = nn.Parameter(torch.randn(r, base.in_features, device=dev) / math.sqrt(base.in_features)); self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
    on = [True]
    def forward(self, x): return self.base(x) + (((x.float() @ self.A.t()) @ self.B.t()).to(x.dtype) if LoRA.on[0] else 0)
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
clips = sorted(d for d in glob.glob(a.data + "/*_s*") if os.path.exists(d + "/meta.json")); random.Random(1).shuffle(clips); VAL, TRAIN = clips[:a.val], clips[a.val:]
print("[rft] clips %d train / %d val, LoRA %.1fM params" % (len(TRAIN), len(VAL), sum(p.numel() for p in params) / 1e6), flush=True)
def gen(seed, c, k): return torch.Generator().manual_seed(seed * 1009 + c * 7 + 31 * k)
def pred(z, k, ci, grad):
    with torch.set_grad_enabled(grad), torch.amp.autocast("cuda", dtype=b.param_dtype):
        out = m(x=[z.to(b.param_dtype)], t=torch.full((1,), float(b._timesteps[k]), device=dev), cross_attn_first_call=not b._cross_init["target"], **b._kwargs("target", ci))[0]
    b._cross_init["target"] = True; return z - sig[k] * out.float()
def err(xh, x0): return ((xh - x0) ** 2).mean() / (x0 ** 2).mean().clamp_min(1e-6)
def clip_iter(d):
    meta = json.load(open(d + "/meta.json")); T = TT[meta["task"]]; seed = meta["seed"]; L = torch.load(d + "/lat.pt").float().to(dev); X0 = torch.load(d + "/x0.pt"); N = meta["chunks"]; XL = torch.load(d + "/x0_loser.pt") if a.dpo > 0 else None
    b.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=meta["max_area"]); b.reset_caches()
    with torch.no_grad(): b.commit(L[:, :3], 0)
    for c in range(1, N):
        cm = meta["chunks_meta"][c]; k = cm["kept"]
        if a.only_flagged is not None and (max(cm["logits"]) - min(cm["logits"])) < a.only_flagged:
            with torch.no_grad(): b.commit(L[:, 3 * c:3 * c + 3], c)
            continue
        x00, x01 = X0[0][c].float().to(dev), X0[1][c].float().to(dev); g = gen(seed, c, k); z0 = b.chunk_noise(c, g); z1 = (1 - sig[1]) * x00 + sig[1] * b.randn_like(x00, g)
        if a.dpo > 0:
            kw = int(max(range(len(cm["logits"])), key=lambda i: cm["logits"][i])); xl0, xl1 = XL[0][c - 1].float().to(dev), XL[1][c - 1].float().to(dev); gl = gen(seed, c, kw); zl0 = b.chunk_noise(c, gl); zl1 = (1 - sig[1]) * xl0 + sig[1] * b.randn_like(xl0, gl)
            yield c, (z0, z1, x00, x01), (zl0, zl1, xl0, xl1)
        else: yield c, z0, z1, x00, x01
        with torch.no_grad(): b.commit(L[:, 3 * c:3 * c + 3], c)
@torch.no_grad()
def validate():
    e = []
    for d in VAL:
        for item in clip_iter(d):
            if a.dpo > 0: c, (z0, z1, x00, x01), (zl0, zl1, xl0, xl1) = item
            else: c, z0, z1, x00, x01 = item
            e.append(float(err(pred(z0, 0, c, False), x00)) + float(err(pred(z1, 1, c, False), x01)))
    return float(np.mean(e))
log = open(os.path.join(a.out, "log.txt"), "a"); step = 0; t0 = time.time(); hist = []; v = validate(); msg = "step 0 val %.4f" % v; print("[rft]", msg, flush=True); log.write(msg + "\n"); best = v
def save(tag): torch.save(dict(rank=a.rank, ffn=a.ffn, step=step, A=[l.A.detach().cpu() for l in LOR], B=[l.B.detach().cpu() for l in LOR]), os.path.join(a.out, "rft_lora%s.pt" % tag))
for ep in range(a.epochs):
    random.shuffle(TRAIN)
    for d in TRAIN:
        for item in clip_iter(d):
            if a.dpo > 0:
                c, W, Lo = item
                def E(pack, grad):
                    z0, z1, x0, x1 = pack; return err(pred(z0, 0, c, grad), x0) + err(pred(z1, 1, c, grad), x1)
                LoRA.on[0] = False
                with torch.no_grad(): ew_ref, el_ref = float(E(W, False)), float(E(Lo, False))
                LoRA.on[0] = True; ew, el = E(W, True), E(Lo, True)
                loss = -torch.nn.functional.logsigmoid(-a.dpo * ((ew - ew_ref) - (el - el_ref))) + 0.1 * ew   # prefer the winner, keep the winner fit
            else:
                c, z0, z1, x00, x01 = item; loss = err(pred(z0, 0, c, True), x00) + err(pred(z1, 1, c, True), x01)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); step += 1; hist.append(float(loss))
            if step % 50 == 0: msg = "step %d ep %d train %.4f  %.0f min" % (step, ep, np.mean(hist[-50:]), (time.time() - t0) / 60); print("[rft]", msg, flush=True); log.write(msg + "\n"); log.flush()
            if a.max_steps and step >= a.max_steps: break
        if step % a.val_every < 12 and step > 0:
            v = validate(); msg = "step %d val %.4f" % (step, v); print("[rft]", msg, flush=True); log.write(msg + "\n"); save("_s%d" % step)
            if v < best: best = v; save("")
        if a.max_steps and step >= a.max_steps: break
    if a.max_steps and step >= a.max_steps: break
v = validate(); msg = "final step %d val %.4f" % (step, v); print("[rft]", msg, flush=True); log.write(msg + "\n"); save("_last")
if v < best: save("")
print("[RFT_TRAIN_DONE] steps %d, %.0f min" % (step, (time.time() - t0) / 60), flush=True)
