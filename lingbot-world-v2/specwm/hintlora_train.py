"""Stage 1: Base-hint LoRA for Fast with deterministic pairing.
Sample (pair_gen.py): teacher-forced prefix (Base rollout latents), fixed chunk noise z0, Base's one-step x0 at sigma~1 (hint0),
the teacher's noisy state at the first low-expert step (z1, sig1), and the teacher's deterministic final x0 (target).
Fast (+LoRA, active only at t >= --lora-tmin so nodes 2/3 stay untouched) gets the HINT through the y channel of the current chunk
(mask value --hint-mask on the chunk frames, image-latent channels = hint0) and must map z0 -> target (node 0) and z1 -> target (node 1).
The target is a deterministic function of (z, prefix) -> MSE does not average over samples (no mean collapse by construction).
torchrun --nproc_per_node 2 -m specwm.hintlora_train --out DIR"""
import argparse, os, sys, glob, json, time, math, random, numpy as np, torch, torch.nn as nn
import torch.distributed as dist
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--pairs", default=S + "/lingbot-world-v2/specwm_out/pairs"); ap.add_argument("--cache", default=S + "/lingbot-world-v2/specwm_out/mixlora/cache")
ap.add_argument("--rank", type=int, default=32); ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--save-every", type=int, default=25)
ap.add_argument("--lora-tmin", type=float, default=900.0, help="LoRA active only for timesteps >= this (nodes 0/1 of the 4-node schedule)")
ap.add_argument("--hint-mask", type=float, default=0.5, help="value written into the y mask channels of the current chunk to mark 'hint frames'")
ap.add_argument("--plan-hint", default="", help="'blur:<scale>' — instead of chunk pairs, the hint is the GT chunk latent downsampled by <scale> (linear) and upsampled back (a stand-in for the low-res whole-clip Base plan); targets = GT chunks; z0 = fresh noise")
ap.add_argument("--nodes", default="0,1"); ap.add_argument("--time-it", type=int, default=0); ap.add_argument("--resume", default=""); ap.add_argument("--chunk0", type=int, default=0, help="1: also train chunk 0 with the hint (frame 0 = the image latent as in y, frames 1-2 = the plan), empty cache — the inference --start 0 situation"); ap.add_argument("--hint-src", default="blur", help="blur: latent-space down/up of the target (plan_blur) | lo: hints/<v>.pt hint_lo bilinearly upsampled | pix: hints/<v>.pt hint_pix (VAE of the cubic-upscaled low-res video)"); ap.add_argument("--hints", default=S + "/lingbot-world-v2/specwm_out/mixlora/hints"); ap.add_argument("--holdout", default="S05_s9999,S01_s9999")
a = ap.parse_args()
dist.init_process_group("nccl"); rank = dist.get_rank(); world = dist.get_world_size(); torch.cuda.set_device(rank); dev = torch.device("cuda", rank)
os.makedirs(a.out, exist_ok=True); random.seed(3000 + rank); torch.manual_seed(3000 + rank)
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding
b = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), device=str(dev))
m = b.target_model.eval(); CS = 3; NODES = [int(x) for x in a.nodes.split(",")]; sig = b.node_sigmas()
for p in m.parameters(): p.requires_grad_(False)
ACTIVE = [True]
class LoRA(nn.Module):
    def __init__(self, base, r):
        super().__init__(); self.base = base; self.A = nn.Parameter(torch.randn(r, base.in_features, device=dev) / math.sqrt(base.in_features)); self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
    def forward(self, x):
        y = self.base(x)
        return y + ((x.float() @ self.A.t()) @ self.B.t()).to(y.dtype) if ACTIVE[0] else y
LOR = []
for blk in m.blocks:
    sa = blk.self_attn
    for nm in ("q", "k", "v", "o"):
        l = LoRA(getattr(sa, nm), a.rank); setattr(sa, nm, l); LOR.append(l)
    for i in (0, 2):
        l = LoRA(blk.ffn[i], a.rank); blk.ffn[i] = l; LOR.append(l)
# the hint enters through patch_embedding's y channels: fine-tune the whole (small) patch embedding
PE = m.patch_embedding; PE.weight.requires_grad_(True); PE.bias.requires_grad_(True); PE0 = {"w": PE.weight.detach().clone(), "b": PE.bias.detach().clone()}
class GatedPE(nn.Module):   # fine-tuned patch embedding only while the hint is active; the original weights otherwise (commit pass, nodes 2/3)
    def __init__(self, pe): super().__init__(); self.pe = pe
    @property
    def weight(self): return self.pe.weight
    @property
    def bias(self): return self.pe.bias
    def forward(self, x):
        if ACTIVE[0]: return self.pe(x)
        return torch.nn.functional.conv3d(x, PE0["w"].to(x.dtype), PE0["b"].to(x.dtype), stride=self.pe.stride, padding=self.pe.padding)
m.patch_embedding = GatedPE(PE)
params = [p for l in LOR for p in (l.A, l.B)] + [PE.weight, PE.bias]
from torch.utils.checkpoint import checkpoint
for blk in m.blocks:
    _f = blk.forward
    def _ck(*args, _f=_f, **kw): return checkpoint(_f, *args, use_reentrant=False, **kw) if torch.is_grad_enabled() else _f(*args, **kw)
    blk.forward = _ck
if a.resume:
    st = torch.load(a.resume)
    for l, A_, B_ in zip(LOR, st["A"], st["B"]): l.A.data.copy_(A_.to(dev)); l.B.data.copy_(B_.to(dev))
    PE.weight.data.copy_(st["pe_w"].to(dev)); PE.bias.data.copy_(st["pe_b"].to(dev))
opt = torch.optim.AdamW([{"params": [p for l in LOR for p in (l.A, l.B)], "lr": a.lr}, {"params": [PE.weight, PE.bias], "lr": a.lr * 0.2}], weight_decay=0.0)
for p in params: dist.broadcast(p.data, 0)
def save(step): torch.save(dict(rank=a.rank, step=step, lora_tmin=a.lora_tmin, hint_mask=a.hint_mask, A=[l.A.detach().cpu() for l in LOR], B=[l.B.detach().cpu() for l in LOR], pe_w=PE.weight.detach().cpu(), pe_b=PE.bias.detach().cpu()), os.path.join(a.out, "hint_lora.pt"))

SC = {sc: dict(img=S + "/lingbot-world-v2/examples/%s/image.jpg" % sc, prompt=open(S + "/p_%s.txt" % sc).read().strip(), act=S + "/rt2_action_%s" % sc) for sc in ("00", "01", "02", "05")}
TASKS = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
def cond_of(sc):
    if sc in SC: return SC[sc]["img"], SC[sc]["prompt"], SC[sc]["act"]
    t = TASKS[sc]; return t["image"], t.get("camera_WM", ""), t["action"]
HOLD = set(a.holdout.split(","))
def scan_pairs():
    P = {}
    for f in sorted(glob.glob(a.pairs + "/*.pt")):
        v, c = os.path.basename(f)[:-3].rsplit("_c", 1); P.setdefault(v, {})[int(c)] = f
    return P, [v for v in sorted(P) if v not in HOLD and len(P[v]) >= 4]
PAIRS, VIDS = scan_pairs()
if a.plan_hint:   # every cached video, every chunk
    VIDS = [os.path.basename(f)[:-3] for f in sorted(glob.glob(a.cache + "/*.pt")) if os.path.basename(f)[:-3] not in HOLD]; PAIRS = {v: {c: None for c in range(1, 13)} for v in VIDS}
def plan_blur(x, scale):
    """[16,f,h,w] -> down by `scale` (bilinear, antialias) -> back up: the resolution loss of a low-res plan"""
    C, f, h, w = x.shape; hh, ww = max(2, int(round(h * scale))), max(2, int(round(w * scale)))
    y = torch.nn.functional.interpolate(x.permute(1, 0, 2, 3), size=(hh, ww), mode="bilinear", antialias=True, align_corners=False)
    return torch.nn.functional.interpolate(y, size=(h, w), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
if rank == 0: print("[hintlora] %d videos with pairs (%d held out), LoRA params %.1fM + patch_embedding %.2fM" % (len(VIDS), len([v for v in PAIRS if v in HOLD]), sum(p.numel() for l in LOR for p in (l.A, l.B)) / 1e6, (PE.weight.numel() + PE.bias.numel()) / 1e6), flush=True)

def with_hint(ci, hint, f0=0):
    """y of chunk ci with the hint in its image-latent channels and --hint-mask in the mask channels (restored afterwards); f0=1 keeps latent frame 0 (the image) as is"""
    y = b._y_chunks[ci]; y_old = y.clone(); y[:4, f0:] = a.hint_mask; y[4:, f0:] = hint[:, f0:].to(y.dtype); return y_old
def fast_pred(z, k, ci):
    s = float(sig[k]); b.set_phase("target"); ACTIVE[0] = float(b._timesteps[k]) >= a.lora_tmin
    with torch.enable_grad(), torch.amp.autocast("cuda", dtype=b.param_dtype):
        out = m(x=[z.to(b.param_dtype)], t=torch.full((1,), float(b._timesteps[k]), device=dev), cross_attn_first_call=not b._cross_init["target"], **b._kwargs("target", ci))[0]
    b._cross_init["target"] = True; ACTIVE[0] = True
    return z - s * out.float()

log = open(os.path.join(a.out, "log_rank%d.txt" % rank), "a"); t_start = time.time(); step = 0; ema = None; stop = False
for ep in range(a.epochs):
    if not a.plan_hint: PAIRS, VIDS = scan_pairs()
    order = list(VIDS); random.Random(ep).shuffle(order); mine = order[rank::world]   # the pair set grows while pair_gen runs
    if rank == 0: print("[hintlora] epoch %d: %d videos, %d pairs" % (ep, len(VIDS), sum(len(PAIRS[v]) for v in VIDS)), flush=True)
    for v in mine:
        D = torch.load(os.path.join(a.cache, v + ".pt")); lat = D["lat"].to(dev).float(); img, prompt, act = cond_of(D["scene"]); HV = None
        if a.plan_hint and a.hint_src != "blur":
            hp = os.path.join(a.hints, v + ".pt")
            if not os.path.exists(hp): continue
            Hd = torch.load(hp); HV = Hd["hint_pix"].to(dev).float() if a.hint_src == "pix" else torch.nn.functional.interpolate(Hd["hint_lo"].to(dev).float().permute(1, 0, 2, 3), size=lat.shape[2:], mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
        b.prepare(img, prompt, act, frame_num=161, max_area=480 * 832); b.reset_caches()
        if not a.chunk0:
            with torch.no_grad(): b.commit(lat[:, :CS], 0)
        for ci in range(0 if a.chunk0 else 1, lat.shape[1] // CS):
            x0 = lat[:, ci * CS:(ci + 1) * CS]
            if ci in PAIRS[v] or (ci == 0 and a.chunk0 and a.plan_hint):
                if a.plan_hint:
                    tgt = x0; sc = float(a.plan_hint.split(":")[1]); hint = HV[:, ci * CS:(ci + 1) * CS] if HV is not None else plan_blur(tgt, sc); P = None
                else:
                    try: P = torch.load(PAIRS[v][ci])
                    except Exception: continue   # file still being written
                    tgt = P["target"].to(dev).float(); hint = P["hint0"].to(dev).float()
                t0 = time.time(); losses = []
                y_old = with_hint(ci, hint, f0=1 if ci == 0 else 0)
                for k in NODES:
                    if P is not None and k == 0: z = P["z0"].to(dev).float()
                    elif P is not None and k == 1: z = P["z1"].to(dev).float()
                    else: s = float(sig[k]); z = (1 - s) * tgt + s * torch.randn_like(tgt)
                    pred = fast_pred(z, k, ci); loss = ((pred - tgt) ** 2).mean() / (tgt ** 2).mean().clamp_min(1e-6)
                    opt.zero_grad(set_to_none=True); loss.backward()
                    for p in params:
                        if p.grad is not None: dist.all_reduce(p.grad); p.grad /= world
                    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); losses.append(loss.item())
                b._y_chunks[ci].copy_(y_old)
                step += 1; lv = float(np.mean(losses)); ema = lv if ema is None else 0.98 * ema + 0.02 * lv
                msg = "ep %d step %d %s c%d loss %.4f (%s) ema %.4f %.1fs %.0fm" % (ep, step, v, ci, lv, " ".join("%.3f" % x for x in losses), ema, time.time() - t0, (time.time() - t_start) / 60)
                log.write(msg + "\n"); log.flush()
                if rank == 0 and (step <= 3 or step % 10 == 0): print("[hintlora]", msg, flush=True)
                if rank == 0 and step % a.save_every == 0: save(step)
            with torch.no_grad(): b.commit(x0, ci)   # teacher forcing with Base's own chunk (the hint must not leak into the cache: y restored above)
            if a.time_it and step >= a.time_it: stop = True; break
        if stop: break
    if stop: break
if rank == 0: save(step)
dist.destroy_process_group()
