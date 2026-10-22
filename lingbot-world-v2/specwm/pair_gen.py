"""Base-only analysis (offline, one GPU).
A. prefix ablation: which parts of the prefix Base actually uses when it RePaints chunk ci.
   prefix = VAE-encoded Base 70-step rollout frames; variants transform the OLD part of the prefix (between the sink chunk 0 and the last 12 latents):
   full (seed 0 / seed 1 = sampling-noise floor), blur2 / blur4 (old latents downsampled 2x / 4x and back = lower-res old prefix), drop_old (old latents not enforced = Base does not see them).
B. step commitment: for the full/seed-0 run, Base's x0 prediction of chunk ci at every step, compared with its final result and with Base's own continuation (GT).
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.base_probe --out DIR [--tasks castle:4321,jar:1234] [--chunks 6,10]"""
import argparse, os, sys, json, time, zlib, numpy as np, torch, imageio, cv2
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--tasks", default=""); ap.add_argument("--chunks", default="1,2,3,4,5,6,7,8,9,10,11,12")
ap.add_argument("--cache", default=S + "/lingbot-world-v2/specwm_out/mixlora/cache"); ap.add_argument("--videos", default="", help="comma list of cache basenames (without .pt); '' = all"); ap.add_argument("--shard", default="0/1", help="i/n: take every n-th video starting at i")
ap.add_argument("--cfg-steps", type=int, default=6); ap.add_argument("--reuse", default="9,11,13")
ap.add_argument("--lora", default="", help="mixed-memory LoRA dir (lora_high.pt, lora_low.pt) for mixL<K>_r<R> variants")
ap.add_argument("--variants", default="full_s0,full_s1,blur2,blur4,drop_old"); ap.add_argument("--steps", type=int, default=16); ap.add_argument("--shift", type=float, default=10.0); ap.add_argument("--guide", type=float, default=5.0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world"); os.chdir(S + "/lingbot-world")
import importlib, importlib.util
spec = importlib.util.spec_from_file_location("specwm_v1", S + "/lingbot-world/specwm/__init__.py", submodule_search_locations=[S + "/lingbot-world/specwm"]); pkg = importlib.util.module_from_spec(spec); sys.modules["specwm_v1"] = pkg; spec.loader.exec_module(pkg)
V1Binding = importlib.import_module("specwm_v1.binding_v1").V1Binding
dev = torch.device("cuda:0"); b1 = V1Binding(S + "/lingbot-world/lingbot-world-base-cam", t5_cpu=True, device=str(dev))
live = b1.pipe.high_noise_model.to(dev); b1.device = dev; cur = ["high"]
W = {"high": {n: torch.empty(p.shape, dtype=p.dtype, device="cpu", pin_memory=True).copy_(p) for n, p in live.named_parameters()},
     "low": {n: torch.empty(p.shape, dtype=p.dtype, device="cpu", pin_memory=True).copy_(p) for n, p in b1.pipe.low_noise_model.named_parameters()}}
b1.pipe.low_noise_model = None; torch.cuda.empty_cache()
ML = {}
if a.lora:
    sys.path.insert(0, S + "/lingbot-world-v2"); from specwm.base_mixed import attach_lora, load_lora
    ML["st"] = {ex: torch.load(os.path.join(a.lora, "lora_%s.pt" % ex)) for ex in ("high", "low")}; ML["mods"] = attach_lora(live, ML["st"]["high"])
    print("[probe] lora steps high %d low %d" % (ML["st"]["high"]["step"], ML["st"]["low"]["step"]), flush=True)
def expert(name):
    if cur[0] != name:
        with torch.no_grad():
            for n, p in live.named_parameters(): p.copy_(W[name][n.replace(".base.", ".")], non_blocking=True)
            if ML: load_lora(ML["mods"], ML["st"][name])
        torch.cuda.synchronize(dev); cur[0] = name
    return live
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
vae = b1.pipe.vae; TASKS = json.load(open(S + "/data/stevo/tasks_camera.json")); FULL = {"castle": "sandcastle_wave_impact", "jar": "sand_filling_up_jar"}
torch.hub.set_dir(S + "/.torch_hub"); dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval().to(dev)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
@torch.no_grad()
def feat(fr):   # [3,T,H,W] in [-1,1] -> unit DINO CLS (mean over frames)
    x = ((fr.permute(1, 0, 2, 3).float() + 1) / 2).clamp(0, 1); x = torch.nn.functional.interpolate(x, size=(224, 392), mode="bilinear", antialias=True)
    f = torch.nn.functional.normalize(dino((x - MEAN) / STD).float(), dim=-1).mean(0); return torch.nn.functional.normalize(f, dim=0)
def sv(fr):   # mean HSV saturation / value (0-255)
    u = ((fr.permute(1, 2, 3, 0).float() + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()
    return np.mean([cv2.cvtColor(im, cv2.COLOR_RGB2HSV)[..., 1:].reshape(-1, 2).mean(0) for im in u[::3]], 0)
def dist(fa, sa, fb, sb): return round(float(1 - fa @ fb), 4), round(float(np.abs(sa - sb).sum() / 80), 4)
@torch.no_grad()
def dec(z): return vae.decode([z])[0].clamp(-1, 1)
@torch.no_grad()
def repaint(c, pref, keep, seed, record=False, cfg_steps=-1, reuse=(), nocam=False):
    """16-step RePaint of the 3 latents after pref (CFG); prefix latents with keep=True are enforced each step. record: x0 of the new chunk at every step."""
    L = pref.shape[1]; sch = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False); sch.set_timesteps(a.steps, device=dev, shift=a.shift)
    g = torch.Generator(device=dev).manual_seed(seed); lat = torch.randn((16, L + 3) + tuple(pref.shape[2:]), generator=g, device=dev); km = keep.view(1, -1, 1, 1).to(dev); rec = []
    for i, t in enumerate(sch.timesteps):
        s = float(sch.sigmas[i]); tv = float(t); nz = (1 - s) * pref + s * torch.randn(pref.shape, generator=g, device=dev)
        lat[:, :L] = torch.where(km, nz, lat[:, :L]); m = expert("high" if tv >= b1.boundary_t else "low"); tt = torch.full((1,), tv, device=dev)
        if i in reuse: pass   # keep the previous flow (flow caching)
        else:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                dc = None if nocam else c["dit_cond"]
                fc = m([lat.to(torch.bfloat16)], t=tt, context=[c["ctx"][0]], seq_len=c["seq_len"], y=[c["y"]], dit_cond_dict=dc)[0].float()
                if cfg_steps < 0 or i < cfg_steps:
                    fu = m([lat.to(torch.bfloat16)], t=tt, context=c["ctxn"], seq_len=c["seq_len"], y=[c["y"]], dit_cond_dict=dc)[0].float(); flow = fu + a.guide * (fc - fu)
                else: flow = fc
        if record: rec.append((i, round(tv), "high" if tv >= b1.boundary_t else "low", (lat - s * flow)[:, L:].clone(), lat[:, L:].clone()))
        lat = sch.step(flow.unsqueeze(0), t, lat.unsqueeze(0), return_dict=False, generator=g)[0].squeeze(0)
    return lat[:, L:], rec
@torch.no_grad()
def repaint_mixed(c, pref, K, hi, seed, lora=False):
    """same RePaint, but through base_mixed.mixed_forward: old latents [3:hi] given as K x fewer tokens with strided RoPE (the real speed-up path)."""
    sys.path.insert(0, S + "/lingbot-world-v2"); from specwm.base_mixed import mixed_forward, down, down_cam
    L = pref.shape[1]; cam = c["dit_cond"]["c2ws_plucker_emb"][0]; y = c["y"]; old = down(pref[:, 3:hi], K)
    segs0 = [dict(y=y[:, :3], f0=0, k=1, cam=cam[:, :, :3]), dict(y=down(y[:, 3:hi], K), f0=3, k=K, cam=down_cam(cam[:, :, 3:hi], K), lora=lora), dict(y=y[:, hi:L + 3], f0=hi, k=1, cam=cam[:, :, hi:L + 3])]
    sch = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False); sch.set_timesteps(a.steps, device=dev, shift=a.shift)
    g = torch.Generator(device=dev).manual_seed(seed); lat = torch.randn((16, L + 3) + tuple(pref.shape[2:]), generator=g, device=dev); z = lat[:, L:].clone()
    for i, t in enumerate(sch.timesteps):
        s = float(sch.sigmas[i]); tv = float(t); eps = torch.randn(pref.shape, generator=g, device=dev); nz = (1 - s) * pref + s * eps
        no = (1 - s) * old + s * K * down(eps[:, 3:hi], K)   # same noise stream as repaint(); K x rescales the averaged noise back to unit std
        segs = [dict(segs0[0], x=nz[:, :3]), dict(segs0[1], x=no), dict(segs0[2], x=torch.cat([nz[:, hi:], z], 1))]
        m = expert("high" if tv >= b1.boundary_t else "low")
        fc = mixed_forward(m, segs, tv, [c["ctx"][0]], [2])[0][:, -3:]; fu = mixed_forward(m, segs, tv, c["ctxn"], [2])[0][:, -3:]
        z = sch.step((fu + a.guide * (fc - fu)).unsqueeze(0), t, z.unsqueeze(0), return_dict=False, generator=g)[0].squeeze(0)
    return z
@torch.no_grad()
def repaint_sub(c, pref, K, hi, seed, temporal):
    sys.path.insert(0, S + "/lingbot-world-v2"); from specwm.base_mixed import mixed_forward
    L = pref.shape[1]; cam = c["dit_cond"]["c2ws_plucker_emb"][0]; y = c["y"]
    if temporal: olds = [dict(f0=f, k=1, sub=1, y=y[:, f:f + 1], cam=cam[:, :, f:f + 1], sl=(f, f + 1)) for f in range(3, hi, K)]
    else: olds = [dict(f0=3, k=K, sub=K, y=y[:, 3:hi], cam=cam[:, :, 3:hi], sl=(3, hi))]
    segs0 = [dict(y=y[:, :3], f0=0, k=1, cam=cam[:, :, :3], sl=(0, 3))] + olds + [dict(y=y[:, hi:L + 3], f0=hi, k=1, cam=cam[:, :, hi:L + 3])]
    sch = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False); sch.set_timesteps(a.steps, device=dev, shift=a.shift)
    g = torch.Generator(device=dev).manual_seed(seed); lat = torch.randn((16, L + 3) + tuple(pref.shape[2:]), generator=g, device=dev); z = lat[:, L:].clone()
    for i, t in enumerate(sch.timesteps):
        s = float(sch.sigmas[i]); tv = float(t); nz = (1 - s) * pref + s * torch.randn(pref.shape, generator=g, device=dev)
        segs = [dict(sg, x=nz[:, sg["sl"][0]:sg["sl"][1]]) for sg in segs0[:-1]] + [dict(segs0[-1], x=torch.cat([nz[:, hi:], z], 1))]
        m = expert("high" if tv >= b1.boundary_t else "low"); last = len(segs) - 1
        fc = mixed_forward(m, segs, tv, [c["ctx"][0]], [last])[0][:, -3:]; fu = mixed_forward(m, segs, tv, c["ctxn"], [last])[0][:, -3:]
        z = sch.step((fu + a.guide * (fc - fu)).unsqueeze(0), t, z.unsqueeze(0), return_dict=False, generator=g)[0].squeeze(0)
    return z
def blur_old(pref, k, lo, hi):
    x = pref.clone(); o = x[:, lo:hi].permute(1, 0, 2, 3); h, w = o.shape[2:]
    x[:, lo:hi] = torch.nn.functional.interpolate(torch.nn.functional.interpolate(o, size=(h // k, w // k), mode="bilinear", antialias=True), size=(h, w), mode="bilinear").permute(1, 0, 2, 3); return x

os.makedirs(a.out, exist_ok=True); import json as _j
SC = {sc: dict(img=S + "/lingbot-world-v2/examples/%s/image.jpg" % sc, prompt=open(S + "/p_%s.txt" % sc).read().strip(), act=S + "/rt2_action_%s" % sc) for sc in ("00", "01", "02", "05")}
TSK = {t["id"]: t for t in _j.load(open(S + "/data/stevo/tasks_camera.json"))}
import glob
files = sorted(glob.glob(a.cache + "/*.pt")); i_, n_ = [int(x) for x in a.shard.split("/")]; files = files[i_::n_]
if a.videos: files = [f for f in files if os.path.basename(f)[:-3] in a.videos.split(",")]
reuse = tuple(int(x) for x in a.reuse.split(",") if x)
for f in files:
    D = torch.load(f); lat = D["lat"].to(dev).float(); sc = D["scene"]; name = os.path.basename(f)[:-3]
    if sc in SC: img, prompt, act = SC[sc]["img"], SC[sc]["prompt"], SC[sc]["act"]
    else: img, prompt, act = TSK[sc]["image"], TSK[sc].get("camera_WM", ""), TSK[sc]["action"]
    for ci in [int(x) for x in a.chunks.split(",")]:
        op = os.path.join(a.out, "%s_c%02d.pt" % (name, ci))
        if os.path.exists(op): continue
        L = 3 * ci; F = 4 * (L + 3 - 1) + 1; c = b1.prepare(img, prompt, act, frame_num=F, max_area=480 * 832); pref = lat[:, :L]
        seed = (zlib.crc32(name.encode()) & 0xffff) * 131 + ci; keep = torch.ones(L, dtype=torch.bool); t0 = time.time()
        # the chunk's initial noise: regenerate exactly as repaint() does (same generator sequence)
        g = torch.Generator(device=dev).manual_seed(seed); lat0 = torch.randn((16, L + 3) + tuple(pref.shape[2:]), generator=g, device=dev); z0 = lat0[:, L:].clone()
        z, rec = repaint(c, pref, keep, seed, record=True, cfg_steps=a.cfg_steps, reuse=reuse)
        st1 = [r for r in rec if r[2] == "low"][0]   # first low-expert step (sigma .943): teacher state for node 1
        torch.save(dict(video=name, scene=sc, chunk=ci, z0=z0.to(torch.bfloat16).cpu(), hint0=rec[0][3].to(torch.bfloat16).cpu(), hint5=rec[5][3].to(torch.bfloat16).cpu(),
                        z1=st1[4].to(torch.bfloat16).cpu(), sig1=round(st1[1] / 1000.0, 3), target=z.to(torch.bfloat16).cpu(), cfg_steps=a.cfg_steps, reuse=a.reuse), op)
        print("[pair] %s c%d  %.0fs" % (name, ci, time.time() - t0), flush=True)
print("[pair] done", flush=True)
