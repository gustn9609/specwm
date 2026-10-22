"""Reward-weighted self-training (RAFT / advantage-weighted regression) of the 1.3B causal-fast draft with a LoRA.
Input: rollout dirs from draft_rollout.py. Per task, advantages A = (r - mean_task) / (std_task + eps) over its seeds; only rollouts with A > 0 are
used, weight w = min(exp(A / beta), wmax). For each selected rollout, teacher-force its own chunks (prefix committed with no grad) and minimise
w * || x0_hat(z_k) - x0 ||^2 / ||x0||^2 at every draft node k, z_k = (1-s_k) x0 + s_k eps  (same loss as fastlora_train.py, 1.3B model, one GPU).
Saves <out>/draft_lora.pt (same format as fast_lora.pt: rank, ffn, step, A, B; merge order = blocks x [q,k,v,o,ffn0,ffn2]).
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.draft_rl_train --rollouts D1,D2 --out DIR [--resume prev/draft_lora.pt] [--epochs 1]"""
import argparse, os, sys, glob, json, time, math, random, collections, numpy as np, torch, torch.nn as nn
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--rollouts", required=True); ap.add_argument("--out", required=True); ap.add_argument("--resume", default="")
ap.add_argument("--rank", type=int, default=16); ap.add_argument("--lr", type=float, default=3e-5); ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--beta", type=float, default=1.0); ap.add_argument("--wmax", type=float, default=3.0)
ap.add_argument("--ffn", type=int, default=1); ap.add_argument("--nodes", default="0,1,2,3"); ap.add_argument("--chunks-per-rollout", type=int, default=6); ap.add_argument("--nodes-per-chunk", type=int, default=2); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); random.seed(a.seed); torch.manual_seed(a.seed); dev = torch.device("cuda:0")
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding
FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
b = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), cfg_name="i2v-1.3B", assets_dir=FAST)
m = b.target_model.eval(); CS = 3; NODES = [int(x) for x in a.nodes.split(",")]; sig = [float(s) for s in b.node_sigmas()]
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
if a.resume:
    st = torch.load(a.resume)
    for l, A_, B_ in zip(LOR, st["A"], st["B"]): l.A.data.copy_(A_.to(dev)); l.B.data.copy_(B_.to(dev))
    print("[rl] resumed from %s (step %s)" % (a.resume, st.get("step")), flush=True)
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
# ---- select rollouts by group-relative advantage
R = collections.defaultdict(list)
for d in a.rollouts.split(","):
    for f in glob.glob(d + "/*.pt"):
        meta = torch.load(f, map_location="cpu"); R[meta["task"]].append((f, meta["reward"], meta["drift"], meta["cam_err"]))
SEL = []
for task, rs in R.items():
    if len(rs) < 2: continue
    r = np.array([x[1] for x in rs]); A = (r - r.mean()) / (r.std() + 1e-6)
    for (f, rew, dr, ce), adv in zip(rs, A):
        if adv > 0: SEL.append((f, min(math.exp(adv / a.beta), a.wmax), task, adv, dr, ce))
print("[rl] tasks %d, rollouts %d, selected %d (mean drift %.2f vs all %.2f; mean cam_err %.1f vs all %.1f)" % (len(R), sum(len(v) for v in R.values()), len(SEL),
      np.mean([x[4] for x in SEL]), np.mean([x[2] for v in R.values() for x in v]), np.mean([x[5] for x in SEL]), np.mean([x[3] for v in R.values() for x in v])), flush=True)
TT = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
def pred_x0(z, k, ci):
    b.set_phase("target")
    with torch.enable_grad(), torch.amp.autocast("cuda", dtype=b.param_dtype):
        out = m(x=[z.to(b.param_dtype)], t=torch.full((1,), float(b._timesteps[k]), device=dev), cross_attn_first_call=not b._cross_init["target"], **b._kwargs("target", ci))[0]
    b._cross_init["target"] = True; return z - sig[k] * out.float()
log = open(os.path.join(a.out, "log.txt"), "a"); step = 0; t0 = time.time(); ema = None
for ep in range(a.epochs):
    random.shuffle(SEL)
    for f, w, task, adv, dr, ce in SEL:
        D = torch.load(f, map_location="cpu"); lat = D["lat"].to(dev).float(); T = TT[task]
        b.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); b.reset_caches()
        with torch.no_grad(): b.commit(lat[:, :CS], 0)
        nC = lat.shape[1] // CS; train_c = set(random.sample(range(1, nC), min(a.chunks_per_rollout, nC - 1)))
        for ci in range(1, nC):
            x0 = lat[:, ci * CS:(ci + 1) * CS]; losses = []
            if ci not in train_c:
                with torch.no_grad(): b.commit(x0, ci)
                continue
            for k in random.sample(NODES, min(a.nodes_per_chunk, len(NODES))):
                z = (1 - sig[k]) * x0 + sig[k] * torch.randn_like(x0); loss = w * ((pred_x0(z, k, ci) - x0) ** 2).mean() / (x0 ** 2).mean().clamp_min(1e-6)
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); losses.append(loss.item() / w)
            with torch.no_grad(): b.commit(x0, ci)
            step += 1; lv = float(np.mean(losses)); ema = lv if ema is None else 0.98 * ema + 0.02 * lv
            if step % 20 == 0: msg = "step %d %s w %.2f loss %.4f ema %.4f %.0fm" % (step, task[:20], w, lv, ema, (time.time() - t0) / 60); print("[rl]", msg, flush=True); log.write(msg + "\n"); log.flush()
torch.save(dict(rank=a.rank, ffn=a.ffn, step=step, A=[l.A.detach().cpu() for l in LOR], B=[l.B.detach().cpu() for l in LOR], selected=len(SEL)), os.path.join(a.out, "draft_lora.pt"))
print("[RL_TRAIN_DONE] steps %d, %.0f min" % (step, (time.time() - t0) / 60), flush=True)
