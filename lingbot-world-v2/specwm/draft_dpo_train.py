"""Flow-DPO of the 1.3B causal-fast draft (LoRA) on rollout pairs from draft_rollout.py (Diffusion-DPO / Flow-DPO form, reference = the same model
with the LoRA switched off). Pairs: within each task, the top-k rollouts by reward vs the bottom-k (winner w, loser l). For a training chunk ci and a
draft node k, both clips are teacher-forced on their OWN prefix; the same noise eps is used for both:
  z = (1 - s_k) x0 + s_k eps,  e(x0) = || x0_hat_theta(z) - x0 ||^2 (normalised),  e_ref likewise with the LoRA off
  loss = -log sigmoid( -beta * [ (e_w - e_ref_w) - (e_l - e_ref_l) ] )
i.e. the policy must reconstruct the winner better (relative to the reference) than the loser; no absolute regression toward either sample.
Saves <out>/draft_lora.pt (merge format of fast_lora.pt). usage: CUDA_VISIBLE_DEVICES=g python -m specwm.draft_dpo_train --rollouts D --out O"""
import argparse, os, sys, glob, json, time, math, random, collections, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--rollouts", required=True); ap.add_argument("--out", required=True); ap.add_argument("--resume", default="")
ap.add_argument("--rank", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--beta", type=float, default=500.0); ap.add_argument("--topk", type=int, default=1)
ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--chunks-per-pair", type=int, default=4); ap.add_argument("--nodes", default="0,1,2"); ap.add_argument("--min-gap", type=float, default=0.3, help="min reward gap for a pair")
ap.add_argument("--ffn", type=int, default=1); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); random.seed(a.seed); torch.manual_seed(a.seed); dev = torch.device("cuda:0")
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding
FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
def mk(): return V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), cfg_name="i2v-1.3B", assets_dir=FAST)
bw, bl = mk(), mk()                       # two bindings = two independent KV caches (winner / loser prefixes); the model weights are shared below
m = bw.target_model.eval(); bl.target_model = m
CS = 3; NODES = [int(x) for x in a.nodes.split(",")]; sig = [float(s) for s in bw.node_sigmas()]
for p in m.parameters(): p.requires_grad_(False)
LORA_ON = [True]
class LoRA(nn.Module):
    def __init__(self, base, r):
        super().__init__(); self.base = base; self.A = nn.Parameter(torch.randn(r, base.in_features, device=dev) / math.sqrt(base.in_features)); self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
    def forward(self, x):
        y = self.base(x)
        return y + ((x.float() @ self.A.t()) @ self.B.t()).to(x.dtype) if LORA_ON[0] else y
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
if a.resume:
    st = torch.load(a.resume)
    for l, A_, B_ in zip(LOR, st["A"], st["B"]): l.A.data.copy_(A_.to(dev)); l.B.data.copy_(B_.to(dev))
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
# ---- pairs
R = collections.defaultdict(list)
for d in a.rollouts.split(","):
    for f in glob.glob(d + "/*.pt"):
        meta = torch.load(f, map_location="cpu"); R[meta["task"]].append((meta["reward"], f, meta["drift"], meta["cam_err"]))
PAIRS = []
for task, rs in R.items():
    rs = sorted(rs, reverse=True)
    for i in range(min(a.topk, len(rs) // 2)):
        w, l = rs[i], rs[-1 - i]
        if w[0] - l[0] >= a.min_gap: PAIRS.append((task, w[1], l[1], w[2] - l[2], w[3] - l[3]))
print("[dpo] tasks %d, pairs %d (winner-loser: drift %+.2f, cam_err %+.1f on average)" % (len(R), len(PAIRS), np.mean([p[3] for p in PAIRS]), np.mean([p[4] for p in PAIRS])), flush=True)
TT = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
def pred(b, z, k, ci, grad):
    b.set_phase("target")
    with torch.set_grad_enabled(grad), torch.amp.autocast("cuda", dtype=b.param_dtype):
        out = m(x=[z.to(b.param_dtype)], t=torch.full((1,), float(b._timesteps[k]), device=dev), cross_attn_first_call=not b._cross_init["target"], **b._kwargs("target", ci))[0]
    b._cross_init["target"] = True; return z - sig[k] * out.float()
def err(xh, x0): return ((xh - x0) ** 2).mean() / (x0 ** 2).mean().clamp_min(1e-6)
log = open(os.path.join(a.out, "log.txt"), "a"); step = 0; t0 = time.time(); acc_hist = []
for ep in range(a.epochs):
    random.shuffle(PAIRS)
    for task, fw, fl, dd, dc in PAIRS:
        T = TT[task]; LW = torch.load(fw, map_location="cpu")["lat"].to(dev).float(); LL = torch.load(fl, map_location="cpu")["lat"].to(dev).float()
        for b in (bw, bl): b.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); b.reset_caches()
        with torch.no_grad(): bw.commit(LW[:, :CS], 0); bl.commit(LL[:, :CS], 0)
        nC = LW.shape[1] // CS; train_c = set(random.sample(range(1, nC), min(a.chunks_per_pair, nC - 1)))
        for ci in range(1, nC):
            xw, xl = LW[:, ci * CS:(ci + 1) * CS], LL[:, ci * CS:(ci + 1) * CS]
            if ci in train_c:
                k = random.choice(NODES); eps = torch.randn_like(xw); s = sig[k]; zw = (1 - s) * xw + s * eps; zl = (1 - s) * xl + s * eps
                with torch.no_grad():
                    LORA_ON[0] = False; rw = err(pred(bw, zw, k, ci, False), xw); rl = err(pred(bl, zl, k, ci, False), xl); LORA_ON[0] = True
                ew = err(pred(bw, zw, k, ci, True), xw); el = err(pred(bl, zl, k, ci, True), xl)
                margin = (ew - rw) - (el - rl); loss = -F.logsigmoid(-a.beta * margin)
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); step += 1; acc_hist.append(float(margin < 0))
                if step % 20 == 0:
                    msg = "step %d node %d loss %.4f margin %+.5f implicit-acc(last 100) %.2f %.0fm" % (step, k, loss.item(), margin.item(), np.mean(acc_hist[-100:]), (time.time() - t0) / 60); print("[dpo]", msg, flush=True); log.write(msg + "\n"); log.flush()
            with torch.no_grad(): bw.commit(xw, ci); bl.commit(xl, ci)
torch.save(dict(rank=a.rank, ffn=a.ffn, step=step, A=[l.A.detach().cpu() for l in LOR], B=[l.B.detach().cpu() for l in LOR], pairs=len(PAIRS)), os.path.join(a.out, "draft_lora.pt"))
print("[DPO_DONE] steps %d, %.0f min" % (step, (time.time() - t0) / 60), flush=True)
