"""Planner adaptation: turn the public 4-step LoRA planner (Base + lightx2v 1022 merged, 576x304, 4 steps shift 5, no CFG) into OUR plan
distribution by a LoRA on both experts trained with x0-regression at the 4 schedule nodes toward reference low-res clips:
plandata hints hint_lo (16-step cr plans of the training set) and mixlora hints hint_lo (Base-70 rollouts downscaled = ideal plans).
z_k = (1-s_k) x0 + s_k eps ; x0_hat = z_k - s_k v(z_k) ; loss = |x0_hat - x0|^2 / |x0|^2. Blur is acceptable for a plan; hallucination is not.
torchrun --nproc_per_node 2 -m specwm.planlora_train --out DIR (one GPU per rank: both experts + LoRA fit in 95 GB at low res)."""
import argparse, os, sys, glob, json, time, math, random, numpy as np, torch, torch.nn as nn, torch.distributed as dist
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--rank", type=int, default=64); ap.add_argument("--lr", type=float, default=3e-5); ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--save-every", type=int, default=25)
ap.add_argument("--area", type=int, default=184320); ap.add_argument("--steps", type=int, default=4); ap.add_argument("--shift", type=float, default=5.0); ap.add_argument("--lora", default="1022"); ap.add_argument("--time-it", type=int, default=0); ap.add_argument("--resume", default="")
ap.add_argument("--max-tokens", type=int, default=28800, help="skip clips whose low-res token count exceeds this (GPU memory)"); ap.add_argument("--nodes", default="1,2,3", help="schedule nodes to train (node 0 = pure noise, high expert: MSE there is mean-seeking; default trains the low-expert nodes)"); ap.add_argument("--sets", default="plandata,mixlora", help="which hint caches to use as targets"); ap.add_argument("--holdout", default="S05_s9999,S01_s9999")
a = ap.parse_args()
NODES = [int(x) for x in a.nodes.split(",")]; NEED_HIGH = 0 in NODES
dist.init_process_group("nccl"); rank = dist.get_rank(); world = dist.get_world_size()
dev = torch.device("cuda", 2 * rank if NEED_HIGH else rank); devB = torch.device("cuda", 2 * rank + 1) if NEED_HIGH else None; torch.cuda.set_device(dev)   # low expert on dev, high expert on devB (both + activations do not fit one GPU)
os.makedirs(a.out, exist_ok=True); random.seed(6000 + rank); torch.manual_seed(6000 + rank)
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world"); os.chdir(S + "/lingbot-world")
import importlib, importlib.util
spec = importlib.util.spec_from_file_location("specwm_v1", S + "/lingbot-world/specwm/__init__.py", submodule_search_locations=[S + "/lingbot-world/specwm"]); pkg = importlib.util.module_from_spec(spec); sys.modules["specwm_v1"] = pkg; spec.loader.exec_module(pkg)
V1Binding = importlib.import_module("specwm_v1.binding_v1").V1Binding
b1 = V1Binding(S + "/lingbot-world/lingbot-world-base-cam", t5_cpu=True, device=str(dev)); EXP = {"low": b1.pipe.low_noise_model.to(dev)}; b1.device = dev; DEV = {"low": dev}
if NEED_HIGH: EXP["high"] = b1.pipe.high_noise_model.to(devB); DEV["high"] = devB
else: b1.pipe.high_noise_model = None; torch.cuda.empty_cache()
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
sys.path.insert(0, S + "/lingbot-world-v2")
LORA = {"1022": ("/home/hk28965/specwm/hf/lightx2v/Wan2.2-Distill-Loras/wan2.2_i2v_A14b_%s_noise_lora_rank64_lightx2v_4step_1022.safetensors", False)}
def merge_lora(Wd, path, use_alpha, scale):
    from safetensors import safe_open
    with safe_open(path, "pt") as fh:
        keys = set(fh.keys())
        for k in keys:
            if not k.endswith(".lora_down.weight"): continue
            base = k[len("diffusion_model."):-len(".lora_down.weight")]; wk = base + ".weight"
            if wk not in Wd: continue
            dn = fh.get_tensor(k).float(); up = fh.get_tensor("diffusion_model." + base + ".lora_up.weight").float()
            sc = scale * (float(fh.get_tensor("diffusion_model." + base + ".alpha")) / dn.shape[0] if use_alpha and ("diffusion_model." + base + ".alpha") in keys else 1.0)
            dW = (up.flatten(1) @ dn.flatten(1)).reshape(Wd[wk].shape).to(Wd[wk].device) * sc; Wd[wk].copy_((Wd[wk].float() + dW).to(Wd[wk].dtype))
        for k in keys:
            if k.endswith(".diff_b") or k.endswith(".diff"):
                base = k[len("diffusion_model."):].rsplit(".", 1)[0]; wk = base + (".bias" if k.endswith(".diff_b") else ".weight")
                if wk in Wd and Wd[wk].shape == fh.get_tensor(k).shape: Wd[wk].copy_((Wd[wk].float() + scale * fh.get_tensor(k).float().to(Wd[wk].device)).to(Wd[wk].dtype))
if a.lora:
    pth, ua = LORA[a.lora]
    with torch.no_grad():
        for ex in EXP: merge_lora(dict(EXP[ex].named_parameters()), pth % ex, ua, 1.0)
for m in EXP.values():
    for p in m.parameters(): p.requires_grad_(False)
class LoRA(nn.Module):
    def __init__(self, base, r):
        super().__init__(); self.base = base; d = base.weight.device; self.A = nn.Parameter(torch.randn(r, base.in_features, device=d) / math.sqrt(base.in_features)); self.B = nn.Parameter(torch.zeros(base.out_features, r, device=d))
    def forward(self, x): y = self.base(x); return y + ((x.float() @ self.A.t()) @ self.B.t()).to(y.dtype)
LOR = {ex: [] for ex in EXP}
for ex, m in EXP.items():
    for blk in m.blocks:
        sa = blk.self_attn
        for nm in ("q", "k", "v", "o"):
            l = LoRA(getattr(sa, nm), a.rank); setattr(sa, nm, l); LOR[ex].append(l)
        for i in (0, 2):
            l = LoRA(blk.ffn[i], a.rank); blk.ffn[i] = l; LOR[ex].append(l)
params = [p for ex in LOR for l in LOR[ex] for p in (l.A, l.B)]
if a.resume:
    st = torch.load(a.resume)
    for ex in LOR:
        for l, A_, B_ in zip(LOR[ex], st["A"][ex], st["B"][ex]): l.A.data.copy_(A_.to(l.A.device)); l.B.data.copy_(B_.to(l.B.device))
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
from torch.utils.checkpoint import checkpoint
for m in EXP.values():
    for blk in m.blocks:
        _f = blk.forward
        def _ck(*args, _f=_f, **kw): return checkpoint(_f, *args, use_reentrant=False, **kw) if torch.is_grad_enabled() else _f(*args, **kw)
        blk.forward = _ck
for p in params: dist.broadcast(p.data, 0)
def save(step): torch.save(dict(rank=a.rank, step=step, steps=a.steps, shift=a.shift, lora=a.lora, area=a.area, nodes=NODES, A={ex: [l.A.detach().cpu() for l in LOR[ex]] for ex in LOR}, B={ex: [l.B.detach().cpu() for l in LOR[ex]] for ex in LOR}), os.path.join(a.out, "plan_lora.pt"))
sch = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False); sch.set_timesteps(a.steps, device="cpu", shift=a.shift); TS = [float(t) for t in sch.timesteps]; SG = [float(s) for s in sch.sigmas[:a.steps]]
if rank == 0: print("[planlora] LoRA %.0fM params on both experts; nodes t=%s sigma=%s" % (sum(p.numel() for p in params) / 1e6, [round(t) for t in TS], [round(s, 3) for s in SG]), flush=True)
SC = {sc: dict(img=S + "/lingbot-world-v2/examples/%s/image.jpg" % sc, prompt=open(S + "/p_%s.txt" % sc).read().strip(), act=S + "/rt2_action_%s" % sc) for sc in ("00", "01", "02", "05")}
TASKS = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
def cond_of(sc):
    if sc in SC: return SC[sc]["img"], SC[sc]["prompt"], SC[sc]["act"]
    t = TASKS[sc]; return t["image"], t.get("camera_WM", ""), t["action"]
HOLD = set(a.holdout.split(",")); ITEMS = []
for st_ in a.sets.split(","):
    for f in sorted(glob.glob(S + "/lingbot-world-v2/specwm_out/%s/hints/*.pt" % st_)):
        v = os.path.basename(f)[:-3]
        if v in HOLD: continue
        ITEMS.append((st_, v, f, S + "/lingbot-world-v2/specwm_out/%s/cache/%s.pt" % (st_, v)))
if rank == 0: print("[planlora] %d target clips (%s)" % (len(ITEMS), a.sets), flush=True)
log = open(os.path.join(a.out, "log_rank%d.txt" % rank), "a"); t_start = time.time(); step = 0; ema = None; stop = False
for ep in range(a.epochs):
    order = list(range(len(ITEMS))); random.Random(ep).shuffle(order); mine = order[rank::world]
    for si in mine:
        st_, v, hf, cf = ITEMS[si]; scene = torch.load(cf)["scene"]; img, prompt, act = cond_of(scene)
        c = b1.prepare(img, prompt, act, frame_num=161, max_area=a.area); x0 = torch.load(hf)["hint_lo"].to(dev).float()
        if c["seq_len"] > a.max_tokens: continue   # e.g. 44x66 grids (29.8k tokens) do not fit next to the expert
        if tuple(c["lat"][1:]) != tuple(x0.shape[2:]): x0 = torch.nn.functional.interpolate(x0.permute(1, 0, 2, 3), size=tuple(c["lat"][1:]), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
        t0 = time.time(); losses = []
        mv = lambda o, d: {k_: mv(v_, d) for k_, v_ in o.items()} if isinstance(o, dict) else ([mv(v_, d) for v_ in o] if isinstance(o, (list, tuple)) else (o.to(d) if torch.is_tensor(o) else o))
        CB = {"low": c}
        if NEED_HIGH: CB["high"] = mv(c, devB)
        for k in NODES:
            s = SG[k]; tv = TS[k]; ex = "high" if tv >= b1.boundary_t else "low"; m = EXP[ex]; d = DEV[ex]; cc = CB[ex]; x0d = x0.to(d); z = (1 - s) * x0d + s * torch.randn_like(x0d)
            with torch.enable_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                vpred = m([z.to(torch.bfloat16)], t=torch.full((1,), tv, device=d), context=[cc["ctx"][0]], seq_len=cc["seq_len"], y=[cc["y"]], dit_cond_dict=cc["dit_cond"])[0].float()
            loss = ((z - s * vpred - x0d) ** 2).mean() / (x0d ** 2).mean().clamp_min(1e-6)
            opt.zero_grad(set_to_none=True); loss.backward()
            for p in params:
                if p.grad is not None: dist.all_reduce(p.grad); p.grad /= world
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); losses.append(loss.item())
        step += 1; lv = float(np.mean(losses)); ema = lv if ema is None else 0.95 * ema + 0.05 * lv
        msg = "ep %d step %d %s/%s loss %.4f (%s) ema %.4f %.1fs %.0fm peak %.0fGB" % (ep, step, st_, v, lv, " ".join("%.3f" % x for x in losses), ema, time.time() - t0, (time.time() - t_start) / 60, max([torch.cuda.max_memory_allocated(dv) for dv in DEV.values()]) / 2**30)
        log.write(msg + "\n"); log.flush()
        if rank == 0 and (step <= 3 or step % 10 == 0): print("[planlora]", msg, flush=True)
        if rank == 0 and step % a.save_every == 0: save(step)
        if a.time_it and step >= a.time_it: stop = True; break
    if stop: break
if rank == 0: save(step)
dist.destroy_process_group()
