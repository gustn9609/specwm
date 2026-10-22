"""A: Fast LoRA distillation from Base rollouts (teacher forcing) — make Fast's 4-node chunk predictions match Base's video.

Per video (Base 70-step rollout latents): walk the chunks in order; for chunk c the Fast cache holds the TEACHER prefix (chunks 0..c-1
committed with the clean t=0 pass, no grad). For each node k: z_k = (1-s_k) x0 + s_k eps, x0 = the teacher chunk; Fast (+LoRA) predicts
x0_hat = z_k - s_k v(z_k); loss = mean (x0_hat - x0)^2 / mean x0^2. Then the teacher chunk is committed and we move on.
LoRA (rank R) on self-attn q/k/v/o and the FFN linears of every block. DDP: torchrun --nproc_per_node 2 (one video per rank per step, grads averaged).
Data: <cache>/*.pt with lat [16,41,h,w] and scene meta (mixlora_cache.py / stevo_cache.py). Held out: jar, castle (not in the cache)."""
import argparse, os, sys, glob, json, time, math, random, numpy as np, torch, torch.nn as nn
import torch.distributed as dist
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--cache", default=S + "/lingbot-world-v2/specwm_out/mixlora/cache")
ap.add_argument("--rank", type=int, default=32); ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--nodes", default="0,1,2,3", help="which Fast nodes to train on each chunk"); ap.add_argument("--save-every", type=int, default=50)
ap.add_argument("--ffn", type=int, default=1); ap.add_argument("--time-it", type=int, default=0); ap.add_argument("--resume", default="")
ap.add_argument("--onpolicy", default="", help="dir of onpolicy_gen.py samples (prefix = Fast's own rollout, target = Base's long-window continuation chunk); mixed into the epoch")
a = ap.parse_args()
dist.init_process_group("nccl"); rank = dist.get_rank(); world = dist.get_world_size(); torch.cuda.set_device(rank); dev = torch.device("cuda", rank)
os.makedirs(a.out, exist_ok=True); random.seed(2000 + rank); torch.manual_seed(2000 + rank)
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding
b = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), device=str(dev))
m = b.target_model.eval(); CS = 3; NODES = [int(x) for x in a.nodes.split(",")]; sig = b.node_sigmas()
for p in m.parameters(): p.requires_grad_(False)

class LoRA(nn.Module):
    def __init__(self, base, r):
        super().__init__(); self.base = base; self.A = nn.Parameter(torch.randn(r, base.in_features, device=dev) / math.sqrt(base.in_features)); self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
    def forward(self, x): return self.base(x) + ((x.float() @ self.A.t()) @ self.B.t()).to(x.dtype)
LOR = []
for blk in m.blocks:
    sa = blk.self_attn
    for nm in ("q", "k", "v", "o"):
        l = LoRA(getattr(sa, nm), a.rank); setattr(sa, nm, l); LOR.append(l)
    if a.ffn:
        for i in (0, 2):
            l = LoRA(blk.ffn[i], a.rank); blk.ffn[i] = l; LOR.append(l)
params = [p for l in LOR for p in (l.A, l.B)]
# gradient checkpointing per block (activations of 40 blocks over 4.6k queries x 28k keys do not fit otherwise)
from torch.utils.checkpoint import checkpoint
import functools
for blk in m.blocks:
    _f = blk.forward
    def _ck(*args, _f=_f, **kw):
        return checkpoint(_f, *args, use_reentrant=False, **kw) if torch.is_grad_enabled() else _f(*args, **kw)
    blk.forward = _ck
if a.resume:
    st = torch.load(a.resume)
    for l, A_, B_ in zip(LOR, st["A"], st["B"]): l.A.data.copy_(A_.to(dev)); l.B.data.copy_(B_.to(dev))
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
for p in params: dist.broadcast(p.data, 0)
if rank == 0: print("[fastlora] LoRA params %.1fM, world %d, nodes %s" % (sum(p.numel() for p in params) / 1e6, world, NODES), flush=True)

FILES = sorted(glob.glob(a.cache + "/*.pt")); assert FILES
ONP = sorted(glob.glob(a.onpolicy + "/*.pt")) if a.onpolicy else []
if rank == 0: print("[fastlora] %d teacher-forcing videos, %d on-policy samples" % (len(FILES), len(ONP)), flush=True)
SC = {sc: dict(img=S + "/lingbot-world-v2/examples/%s/image.jpg" % sc, prompt=open(S + "/p_%s.txt" % sc).read().strip(), act=S + "/rt2_action_%s" % sc) for sc in ("00", "01", "02", "05")}
TASKS = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
def cond_of(D):
    sc = D["scene"]
    if sc in SC: return SC[sc]["img"], SC[sc]["prompt"], SC[sc]["act"]
    t = TASKS[sc]; return t["image"], t.get("camera_WM", ""), t["action"]

def train_chunk(x0, ci, tag):
    """one training step on chunk ci (target x0) with the cache holding the prefix; returns per-node losses"""
    global step, ema
    t0 = time.time(); losses = []
    for k in NODES:
        s = float(sig[k]); z = (1 - s) * x0 + s * torch.randn_like(x0)
        pred = fast_pred(z, k, ci); loss = ((pred - x0) ** 2).mean() / (x0 ** 2).mean().clamp_min(1e-6)
        opt.zero_grad(set_to_none=True); loss.backward()
        for p in params:
            if p.grad is not None: dist.all_reduce(p.grad); p.grad /= world
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); losses.append(loss.item())
    step += 1; lv = float(np.mean(losses)); ema = lv if ema is None else 0.98 * ema + 0.02 * lv
    msg = "step %d %s chunk %d loss %.4f (nodes %s) ema %.4f  %.1fs elapsed %.0fm" % (step, tag, ci, lv, " ".join("%.3f" % x for x in losses), ema, time.time() - t0, (time.time() - t_start) / 60)
    log.write(msg + "\n"); log.flush()
    if rank == 0 and (step <= 3 or step % 10 == 0): print("[fastlora]", msg, flush=True)
    if rank == 0 and step % a.save_every == 0: torch.save(dict(rank=a.rank, ffn=a.ffn, step=step, A=[l.A.detach().cpu() for l in LOR], B=[l.B.detach().cpu() for l in LOR]), os.path.join(a.out, "fast_lora.pt"))
def fast_pred(z, k, ci):
    """Fast x0 prediction at node k for chunk ci, with grad through the LoRA (cache holds the committed prefix)."""
    s = float(sig[k]); b.set_phase("target")
    with torch.enable_grad(), torch.amp.autocast("cuda", dtype=b.param_dtype):
        out = m(x=[z.to(b.param_dtype)], t=torch.full((1,), float(b._timesteps[k]), device=dev), cross_attn_first_call=not b._cross_init["target"], **b._kwargs("target", ci))[0]
    b._cross_init["target"] = True
    return z - s * out.float()

log = open(os.path.join(a.out, "log_rank%d.txt" % rank), "a"); t_start = time.time(); step = 0; ema = None
SAMPLES = [("video", f) for f in FILES] + [("onpol", f) for f in ONP]
stop = False
for ep in range(a.epochs):
    order = list(range(len(SAMPLES))); random.Random(ep).shuffle(order); mine = order[rank::world]
    for si in mine:
        kind, f = SAMPLES[si]; D = torch.load(f)
        if kind == "video":
            lat = D["lat"].to(dev).float(); img, prompt, act = cond_of(D)
            b.prepare(img, prompt, act, frame_num=161, max_area=480 * 832); b.reset_caches()
            with torch.no_grad(): b.commit(lat[:, :CS], 0)
            for ci in range(1, lat.shape[1] // CS):
                x0 = lat[:, ci * CS:(ci + 1) * CS]; train_chunk(x0, ci, "tf:" + os.path.basename(f)[:-3])
                with torch.no_grad(): b.commit(x0, ci)   # teacher forcing: the prefix is Base's own chunk
                if a.time_it and step >= a.time_it: stop = True; break
        else:   # on-policy: prefix = Fast's own rollout latents, target = Base's long-window continuation
            t = TASKS[D["task"]]; b.prepare(t["image"], t.get("camera_WM", ""), t["action"], frame_num=161, max_area=480 * 832); b.reset_caches()
            pref = D["prefix"].to(dev).float(); ci = int(D["chunk"])
            with torch.no_grad():
                for cj in range(ci): b.commit(pref[:, cj * CS:(cj + 1) * CS], cj)
            train_chunk(D["target"].to(dev).float(), ci, "op:" + os.path.basename(f)[:-3])
        if stop or (a.time_it and step >= a.time_it): stop = True; break
    if stop: break
if rank == 0: torch.save(dict(rank=a.rank, ffn=a.ffn, step=step, A=[l.A.detach().cpu() for l in LOR], B=[l.B.detach().cpu() for l in LOR]), os.path.join(a.out, "fast_lora.pt"))
dist.destroy_process_group()
