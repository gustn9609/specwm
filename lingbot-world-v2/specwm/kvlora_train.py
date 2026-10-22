"""KV-translator training (Base -> Fast plan memory): Base's low-noise expert reads the LOW-RES plan once (t = --kv-t) and its post-norm pre-RoPE
keys/values at --layers are mapped by a per-block linear Translator into Fast's K/V space (RoPE at the mapped positions) and prepended to Fast's
KV cache as plan memory for every chunk. Jointly trained with a gated LoRA on Fast (nodes t >= --lora-tmin) by teacher forcing on the
(cache, hints) data: z_k = (1-s_k) x0 + s_k eps, loss = |x0_hat - x0|^2 / |x0|^2 at NODES. --y-hint 1 also writes the plan hint into y (plan6 style).
Rank r: Fast on cuda:2r, Base low expert on cuda:2r+1. torchrun --nproc_per_node 2 with 4 visible GPUs."""
import argparse, os, sys, glob, json, time, math, random, numpy as np, torch, torch.nn as nn, torch.distributed as dist
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--cache", default=S + "/lingbot-world-v2/specwm_out/mixdata/cache"); ap.add_argument("--hints", default=S + "/lingbot-world-v2/specwm_out/mixdata/hints")
ap.add_argument("--rank", type=int, default=32); ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--lr-tr", type=float, default=1e-4); ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--save-every", type=int, default=25)
ap.add_argument("--nodes", default="0,1"); ap.add_argument("--lora-tmin", type=float, default=900.0); ap.add_argument("--layers", default="all"); ap.add_argument("--translator", default="v1", help="v1: per-block full linear (kv1) | v2: per-head linear + low-rank mixing (kv4)"); ap.add_argument("--mem-window", type=int, default=-1, help=">=0: time-local memory = plan frames of chunks [c-P, c+F] at full frame rate (kv4; P/F from --mem-past/--mem-future); -1: whole clip at --mem-stride (kv1)"); ap.add_argument("--mem-past", type=int, default=0); ap.add_argument("--mem-future", type=int, default=1); ap.add_argument("--mem-wstride", type=int, default=1, help="frame stride inside the time-local window"); ap.add_argument("--mem-rank", type=int, default=256); ap.add_argument("--mem-stride", type=int, default=2, help="keep every k-th plan latent frame as memory"); ap.add_argument("--kv-t", type=float, default=0.0)
ap.add_argument("--y-hint", type=int, default=0); ap.add_argument("--hint-mask", type=float, default=0.5); ap.add_argument("--resume", default=""); ap.add_argument("--resume-lora", default="", help="init the Fast LoRA (+PE) from a hint_lora.pt")
ap.add_argument("--holdout", default="S05_s9999,S01_s9999,S05_s9999_pd,S01_s9999_pd"); ap.add_argument("--area-lo", type=int, default=184320); ap.add_argument("--time-it", type=int, default=0)
a = ap.parse_args()
rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1)); dev = torch.device("cuda", 2 * rank); devB = torch.device("cuda", 2 * rank + 1); torch.cuda.set_device(dev)
dist.init_process_group("nccl", device_id=dev)   # bind NCCL to the Fast GPU of this rank (2 GPUs per rank: NCCL must not guess cuda:<rank>)
os.makedirs(a.out, exist_ok=True); random.seed(4000 + rank); torch.manual_seed(4000 + rank)
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding
from specwm.plan_kv import capture_base_kv, plan_rope, Translator, Translator2, set_mem, subsample_frames, window_tokens
b = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), device=str(dev))
m = b.target_model.eval(); CS = 3; NODES = [int(x) for x in a.nodes.split(",")]; sig = b.node_sigmas(); LAYERS = list(range(len(b.target_model.blocks))) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
for p in m.parameters(): p.requires_grad_(False)
# Base low expert (v1 repo) on devB
import importlib, importlib.util
os.chdir(S + "/lingbot-world"); sys.path = [S + "/lingbot-world"] + [p for p in sys.path if p != S + "/lingbot-world-v2"]
for k in [k for k in sys.modules if k == "wan" or k.startswith("wan.")]: del sys.modules[k]   # the v1 repo's own `wan` (as spec_v0.py does)
spec = importlib.util.spec_from_file_location("specwm_v1", S + "/lingbot-world/specwm/__init__.py", submodule_search_locations=[S + "/lingbot-world/specwm"]); pkg = importlib.util.module_from_spec(spec); sys.modules["specwm_v1"] = pkg; spec.loader.exec_module(pkg)
V1Binding = importlib.import_module("specwm_v1.binding_v1").V1Binding
b1 = V1Binding(S + "/lingbot-world/lingbot-world-base-cam", t5_cpu=True, device=str(devB)); LOWB = b1.pipe.low_noise_model.to(devB); b1.pipe.high_noise_model = None; b1.device = devB; torch.cuda.empty_cache()
os.chdir(S + "/lingbot-world-v2")
ACTIVE = [True]
class LoRA(nn.Module):
    def __init__(self, base, r):
        super().__init__(); self.base = base; self.A = nn.Parameter(torch.randn(r, base.in_features, device=dev) / math.sqrt(base.in_features)); self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
    def forward(self, x):
        y = self.base(x); return y + ((x.to(torch.bfloat16) @ self.A.t().to(torch.bfloat16)) @ self.B.t().to(torch.bfloat16)).to(y.dtype) if ACTIVE[0] else y   # bf16 LoRA maths: no fp32 copies of the activations
LOR = []
for blk in m.blocks:
    sa = blk.self_attn
    for nm in ("q", "k", "v", "o"):
        l = LoRA(getattr(sa, nm), a.rank); setattr(sa, nm, l); LOR.append(l)
    for i in (0, 2):
        l = LoRA(blk.ffn[i], a.rank); blk.ffn[i] = l; LOR.append(l)
PE = m.patch_embedding; PE0 = {"w": PE.weight.detach().clone(), "b": PE.bias.detach().clone()}
if a.y_hint: PE.weight.requires_grad_(True); PE.bias.requires_grad_(True)
class GatedPE(nn.Module):
    def __init__(self, pe): super().__init__(); self.pe = pe
    @property
    def weight(self): return self.pe.weight
    @property
    def bias(self): return self.pe.bias
    def forward(self, x):
        if ACTIVE[0] and a.y_hint: return self.pe(x)
        return torch.nn.functional.conv3d(x, PE0["w"].to(x.dtype), PE0["b"].to(x.dtype), stride=self.pe.stride, padding=self.pe.padding)
m.patch_embedding = GatedPE(PE)
TR = Translator2(LAYERS, rank=a.mem_rank, dev=devB) if a.translator == "v2" else Translator(LAYERS, dev=dev)   # v2: translator lives on the Base GPU; only the translated bf16 K/V cross to the Fast GPU
if a.resume:   # translator (+ LoRA/PE unless --resume-lora overrides them below)
    st = torch.load(a.resume)
    for l, A_, B_ in zip(LOR, st["A"], st["B"]): l.A.data.copy_(A_.to(dev)); l.B.data.copy_(B_.to(dev))
    PE.weight.data.copy_(st["pe_w"].to(dev)); PE.bias.data.copy_(st["pe_b"].to(dev)); TR.load_state_dict(st.get("translator_sd", st.get("translator")))
if a.resume_lora:   # LoRA (+ fine-tuned PE) from a hint_lora.pt (y-hint route)
    st = torch.load(a.resume_lora)
    for l, A_, B_ in zip(LOR, st["A"], st["B"]): l.A.data.copy_(A_.to(dev)); l.B.data.copy_(B_.to(dev))
    if a.y_hint: PE.weight.data.copy_(st["pe_w"].to(dev)); PE.bias.data.copy_(st["pe_b"].to(dev))
params = [p for l in LOR for p in (l.A, l.B)] + ([PE.weight, PE.bias] if a.y_hint else []); tparams = list(TR.parameters())
opt = torch.optim.AdamW([{"params": params, "lr": a.lr}, {"params": tparams, "lr": a.lr_tr}], weight_decay=0.0)
from torch.utils.checkpoint import checkpoint
for blk in m.blocks:
    _f = blk.forward
    def _ck(*args, _f=_f, **kw): return checkpoint(_f, *args, use_reentrant=False, **kw) if torch.is_grad_enabled() else _f(*args, **kw)
    blk.forward = _ck
def _bcast(p):   # NCCL is bound to dev; translator params may live on devB -> collective through a dev copy
    if p.data.device == dev: dist.broadcast(p.data, 0)
    else: t = p.data.to(dev); dist.broadcast(t, 0); p.data.copy_(t)
for p in params + tparams: _bcast(p)
def save(step): torch.save(dict(rank=a.rank, step=step, lora_tmin=a.lora_tmin, hint_mask=a.hint_mask, y_hint=a.y_hint, layers=LAYERS, kv_t=a.kv_t, area_lo=a.area_lo, mem_stride=a.mem_stride, translator=a.translator, mem_window=a.mem_window, mem_past=a.mem_past, mem_future=a.mem_future, mem_wstride=a.mem_wstride, mem_rank=a.mem_rank, A=[l.A.detach().cpu() for l in LOR], B=[l.B.detach().cpu() for l in LOR], pe_w=PE.weight.detach().cpu(), pe_b=PE.bias.detach().cpu(), translator_sd={k: v.detach().cpu() for k, v in TR.state_dict().items()}), os.path.join(a.out, "kv_lora.pt"))
if rank == 0: print("[kvlora] LoRA %.1fM + translator %.0fM params, layers %s, y_hint %d" % (sum(p.numel() for p in params) / 1e6, sum(p.numel() for p in tparams) / 1e6, LAYERS, a.y_hint), flush=True)
SC = {sc: dict(img=S + "/lingbot-world-v2/examples/%s/image.jpg" % sc, prompt=open(S + "/p_%s.txt" % sc).read().strip(), act=S + "/rt2_action_%s" % sc) for sc in ("00", "01", "02", "05")}
TASKS = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
def cond_of(sc):
    if sc in SC: return SC[sc]["img"], SC[sc]["prompt"], SC[sc]["act"]
    t = TASKS[sc]; return t["image"], t.get("camera_WM", ""), t["action"]
HOLD = set(a.holdout.split(",")); VIDS = [os.path.basename(f)[:-3] for f in sorted(glob.glob(a.cache + "/*.pt")) if os.path.basename(f)[:-3] not in HOLD]
def fast_pred(z, k, ci):
    s = float(sig[k]); b.set_phase("target"); ACTIVE[0] = float(b._timesteps[k]) >= a.lora_tmin
    with torch.enable_grad(), torch.amp.autocast("cuda", dtype=b.param_dtype):
        out = m(x=[z.to(b.param_dtype)], t=torch.full((1,), float(b._timesteps[k]), device=dev), cross_attn_first_call=not b._cross_init["target"], **b._kwargs("target", ci))[0]
    b._cross_init["target"] = True; ACTIVE[0] = True
    return z - s * out.float()
log = open(os.path.join(a.out, "log_rank%d.txt" % rank), "a"); t_start = time.time(); step = 0; ema = None; stop = False
for ep in range(a.epochs):
    order = list(VIDS); random.Random(ep).shuffle(order); mine = order[rank::world]
    if rank == 0: print("[kvlora] epoch %d: %d videos" % (ep, len(VIDS)), flush=True)
    for v in mine:
        D = torch.load(os.path.join(a.cache, v + ".pt")); lat = D["lat"].to(dev).float(); img, prompt, act = cond_of(D["scene"])
        hp = os.path.join(a.hints, v + ".pt")
        if not os.path.exists(hp): continue
        Hd = torch.load(hp); lo = Hd["hint_lo"].to(devB).float(); HV = Hd["hint_pix"].to(dev).float() if a.y_hint else None
        # plan memory: Base low expert reads the low-res plan once; translated (with grad) per node forward
        c_lo = b1.prepare(img, prompt, act, frame_num=161, max_area=a.area_lo)
        if tuple(c_lo["lat"][1:]) != tuple(lo.shape[2:]):   # old hint caches used a 16-px grid rule; Base's low-res grid differs by a latent or two -> resample
            lo = torch.nn.functional.interpolate(lo.permute(1, 0, 2, 3), size=tuple(c_lo["lat"][1:]), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
        torch.cuda.synchronize(dev); torch.cuda.synchronize(devB)   # the two GPUs exchange memory tensors during training; settle both before the next Base pass
        KVb = capture_base_kv(LOWB, lo, c_lo, a.kv_t, LAYERS, devB); torch.cuda.synchronize(devB); lo_shape = tuple(lo.shape); del c_lo, lo; torch.cuda.empty_cache()
        if a.mem_window < 0: KV = {l: (k.to(dev), vv.to(dev)) for l, (k, vv) in KVb.items()}; del KVb
        else: KV = None   # time-local: sliced per chunk from KVb (kept on devB)
        log.write("[mem] %s devB alloc %.1f reserved %.1f GB | dev alloc %.1f reserved %.1f GB\n" % (v, torch.cuda.memory_allocated(devB) / 2**30, torch.cuda.memory_reserved(devB) / 2**30, torch.cuda.memory_allocated(dev) / 2**30, torch.cuda.memory_reserved(dev) / 2**30)); log.flush()
        grid_lo = (lo_shape[1], lo_shape[2] // 2, lo_shape[3] // 2); grid_hi = (lat.shape[1], lat.shape[2] // 2, lat.shape[3] // 2)
        if a.mem_window < 0:
            KV, _ = subsample_frames(KV, grid_lo, a.mem_stride); mult = plan_rope(m.freqs.to(dev), grid_lo, grid_hi, frame_ids=list(range(0, grid_lo[0], a.mem_stride)))
            MEM = lambda ci: TR(KV, mult)
        else:
            def MEM(ci, P=a.mem_past, F=a.mem_future):
                kvw, fids = window_tokens(KVb, grid_lo, CS * (ci - P), CS * (ci + 1 + F), dev=None, stride=a.mem_wstride)   # stays on devB
                out = TR(kvw, plan_rope(m.freqs.to(devB), grid_lo, grid_hi, frame_ids=fids))
                return {l: (k.to(dev), v.to(dev)) for l, (k, v) in out.items()}   # grads flow back across the device copy
        if step == 0 and rank == 0: print("[kvlora] %s memory, %d blocks, %s" % ("whole-clip stride %d" % a.mem_stride if a.mem_window < 0 else "time-local chunks [c-%d, c+%d]" % (a.mem_past, a.mem_future), len(LAYERS), a.translator), flush=True)
        b.prepare(img, prompt, act, frame_num=161, max_area=480 * 832); b.reset_caches()
        for ci in range(lat.shape[1] // CS):
            x0 = lat[:, ci * CS:(ci + 1) * CS]; t0 = time.time(); losses = []
            y_old = None
            if HV is not None:
                yy = b._y_chunks[ci]; y_old = yy.clone(); f0 = 1 if ci == 0 else 0; hint = HV[:, ci * CS:(ci + 1) * CS]; yy[:4, f0:] = a.hint_mask; yy[4:, f0:] = hint[:, f0:].to(yy.dtype); y_hint = yy.clone()
            for k in NODES:
                if y_old is not None: b._y_chunks[ci].copy_(y_hint if float(b._timesteps[k]) >= a.lora_tmin else y_old)
                s = float(sig[k]); z = (1 - s) * x0 + s * torch.randn_like(x0)
                set_mem(m, MEM(ci)); pred = fast_pred(z, k, ci); loss = ((pred - x0) ** 2).mean() / (x0 ** 2).mean().clamp_min(1e-6)
                opt.zero_grad(set_to_none=True); loss.backward()
                for p in params + tparams:
                    if p.grad is None: continue
                    if p.grad.device == dev: dist.all_reduce(p.grad); p.grad /= world
                    else: g = p.grad.to(dev); dist.all_reduce(g); p.grad.copy_(g / world)
                torch.nn.utils.clip_grad_norm_(params + tparams, 1.0); opt.step(); losses.append(loss.item())
            if y_old is not None: b._y_chunks[ci].copy_(y_old)
            with torch.no_grad(): set_mem(m, MEM(ci)); b.commit(x0, ci)   # the commit pass also sees the plan memory (as at inference)
            step += 1; lv = float(np.mean(losses)); ema = lv if ema is None else 0.98 * ema + 0.02 * lv
            msg = "ep %d step %d %s c%d loss %.4f (%s) ema %.4f %.1fs %.0fm" % (ep, step, v, ci, lv, " ".join("%.3f" % x for x in losses), ema, time.time() - t0, (time.time() - t_start) / 60)
            log.write(msg + "\n"); log.flush()
            if rank == 0 and (step <= 3 or step % 10 == 0): print("[kvlora]", msg, "peak %.0f GB" % (torch.cuda.max_memory_allocated(dev) / 2**30), flush=True)
            if rank == 0 and step % a.save_every == 0: save(step)
            if a.time_it and step >= a.time_it: stop = True; break
        set_mem(m, None); torch.cuda.synchronize(dev); torch.cuda.synchronize(devB); KVb = None; KV = None; torch.cuda.empty_cache()
        if stop: break
    if stop: break
if rank == 0: save(step)
dist.destroy_process_group()
