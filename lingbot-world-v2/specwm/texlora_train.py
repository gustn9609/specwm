"""Texture LoRA (on-policy / DAgger round): a second LoRA on Fast, active only at the LOW-noise nodes (t < --lora-tmax, i.e. nodes 2/3 that see
the plain y), trained on onpolicy2_gen samples: prefix = Fast's own plan-rollout (committed, no grad), node-2 input z = the rollout's actual
state entering node 2, target = Base's RePaint of that chunk given Fast's prefix (plan-conditioned). Node 3 is trained on the chain
z3 = (1-s3) x0_2 + s3 eps with the CURRENT node-2 prediction (on-policy). Loss = |x0_hat - target|^2 / |target|^2.
torchrun --nproc_per_node 2 -m specwm.texlora_train --out DIR [--samples onpol2/samples]"""
import argparse, os, sys, glob, json, time, math, random, numpy as np, torch, torch.nn as nn, torch.distributed as dist
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--samples", default=S + "/lingbot-world-v2/specwm_out/onpol2/samples")
ap.add_argument("--rank", type=int, default=32); ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--save-every", type=int, default=25)
ap.add_argument("--nodes", default="2,3"); ap.add_argument("--lora-tmax", type=float, default=900.0); ap.add_argument("--resume", default=""); ap.add_argument("--time-it", type=int, default=0)
ap.add_argument("--holdout", default="S05_s9999,S01_s9999")
a = ap.parse_args()
dist.init_process_group("nccl"); rank = dist.get_rank(); world = dist.get_world_size(); torch.cuda.set_device(rank); dev = torch.device("cuda", rank)
os.makedirs(a.out, exist_ok=True); random.seed(5000 + rank); torch.manual_seed(5000 + rank)
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding
b = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), device=str(dev))
m = b.target_model.eval(); CS = 3; NODES = [int(x) for x in a.nodes.split(",")]; sig = b.node_sigmas()
for p in m.parameters(): p.requires_grad_(False)
ACTIVE = [False]
class LoRA(nn.Module):
    def __init__(self, base, r):
        super().__init__(); self.base = base; self.A = nn.Parameter(torch.randn(r, base.in_features, device=dev) / math.sqrt(base.in_features)); self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
    def forward(self, x):
        y = self.base(x); return y + ((x.float() @ self.A.t()) @ self.B.t()).to(y.dtype) if ACTIVE[0] else y
LOR = []
for blk in m.blocks:
    sa = blk.self_attn
    for nm in ("q", "k", "v", "o"):
        l = LoRA(getattr(sa, nm), a.rank); setattr(sa, nm, l); LOR.append(l)
    for i in (0, 2):
        l = LoRA(blk.ffn[i], a.rank); blk.ffn[i] = l; LOR.append(l)
params = [p for l in LOR for p in (l.A, l.B)]
if a.resume:
    st = torch.load(a.resume)
    for l, A_, B_ in zip(LOR, st["A"], st["B"]): l.A.data.copy_(A_.to(dev)); l.B.data.copy_(B_.to(dev))
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
from torch.utils.checkpoint import checkpoint
for blk in m.blocks:
    _f = blk.forward
    def _ck(*args, _f=_f, **kw): return checkpoint(_f, *args, use_reentrant=False, **kw) if torch.is_grad_enabled() else _f(*args, **kw)
    blk.forward = _ck
for p in params: dist.broadcast(p.data, 0)
def save(step): torch.save(dict(rank=a.rank, step=step, lora_tmax=a.lora_tmax, A=[l.A.detach().cpu() for l in LOR], B=[l.B.detach().cpu() for l in LOR]), os.path.join(a.out, "tex_lora.pt"))
SC = {sc: dict(img=S + "/lingbot-world-v2/examples/%s/image.jpg" % sc, prompt=open(S + "/p_%s.txt" % sc).read().strip(), act=S + "/rt2_action_%s" % sc) for sc in ("00", "01", "02", "05")}
TASKS = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
def cond_of(sc):
    if sc in SC: return SC[sc]["img"], SC[sc]["prompt"], SC[sc]["act"]
    t = TASKS[sc]; return t["image"], t.get("camera_WM", ""), t["action"]
HOLD = set(a.holdout.split(",")); FILES = [f for f in sorted(glob.glob(a.samples + "/*.pt")) if os.path.basename(f).rsplit("_c", 1)[0] not in HOLD]
if rank == 0: print("[texlora] %d on-policy samples, nodes %s, LoRA active for t < %.0f" % (len(FILES), NODES, a.lora_tmax), flush=True)
def fast_pred(z, k, ci):
    s = float(sig[k]); b.set_phase("target"); ACTIVE[0] = float(b._timesteps[k]) < a.lora_tmax
    with torch.enable_grad(), torch.amp.autocast("cuda", dtype=b.param_dtype):
        out = m(x=[z.to(b.param_dtype)], t=torch.full((1,), float(b._timesteps[k]), device=dev), cross_attn_first_call=not b._cross_init["target"], **b._kwargs("target", ci))[0]
    b._cross_init["target"] = True   # ACTIVE stays as set: the checkpoint recomputation in backward must see the same LoRA state as the forward
    return z - s * out.float()
log = open(os.path.join(a.out, "log_rank%d.txt" % rank), "a"); t_start = time.time(); step = 0; ema = None; stop = False
for ep in range(a.epochs):
    order = list(range(len(FILES))); random.Random(ep).shuffle(order); mine = order[rank::world]
    for si in mine:
        D = torch.load(FILES[si]); ci = int(D["chunk"]); img, prompt, act = cond_of(D["scene"]); tgt = D["target"].to(dev).float(); pref = D["prefix"].to(dev).float(); Z = D["z"].to(dev).float()
        b.prepare(img, prompt, act, frame_num=161, max_area=480 * 832); b.reset_caches()
        ACTIVE[0] = False
        with torch.no_grad():
            for cj in range(ci): b.commit(pref[:, cj * CS:(cj + 1) * CS], cj)   # Fast's own prefix (plain commit pass, as at inference)
        t0 = time.time(); losses = []; x0 = None
        for k in NODES:
            s = float(sig[k])
            if x0 is None: z = D["z_init"].to(dev).float() if "z_init" in D else Z[k]   # first node: the exact latent Base started from (deterministic pairing), else the rollout's state
            else: z = (1 - s) * x0.detach() + s * torch.randn_like(x0)   # then the on-policy chain
            pred = fast_pred(z, k, ci); loss = ((pred - tgt) ** 2).mean() / (tgt ** 2).mean().clamp_min(1e-6)
            opt.zero_grad(set_to_none=True); loss.backward()
            for p in params:
                if p.grad is not None: dist.all_reduce(p.grad); p.grad /= world
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); losses.append(loss.item()); x0 = pred
        step += 1; lv = float(np.mean(losses)); ema = lv if ema is None else 0.98 * ema + 0.02 * lv
        msg = "ep %d step %d %s loss %.4f (%s) ema %.4f %.1fs %.0fm" % (ep, step, os.path.basename(FILES[si])[:-3], lv, " ".join("%.3f" % x for x in losses), ema, time.time() - t0, (time.time() - t_start) / 60)
        log.write(msg + "\n"); log.flush()
        if rank == 0 and (step <= 3 or step % 10 == 0): print("[texlora]", msg, flush=True)
        if rank == 0 and step % a.save_every == 0: save(step)
        if a.time_it and step >= a.time_it: stop = True; break
    if stop: break
if rank == 0: save(step)
dist.destroy_process_group()
