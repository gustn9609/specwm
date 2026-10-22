"""Base-only analysis (offline, one GPU).
A. prefix ablation: which parts of the prefix Base actually uses when it RePaints chunk ci.
   prefix = VAE-encoded Base 70-step rollout frames; variants transform the OLD part of the prefix (between the sink chunk 0 and the last 12 latents):
   full (seed 0 / seed 1 = sampling-noise floor), blur2 / blur4 (old latents downsampled 2x / 4x and back = lower-res old prefix), drop_old (old latents not enforced = Base does not see them).
B. step commitment: for the full/seed-0 run, Base's x0 prediction of chunk ci at every step, compared with its final result and with Base's own continuation (GT).
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.base_probe --out DIR [--tasks castle:4321,jar:1234] [--chunks 6,10]"""
import argparse, os, sys, json, time, numpy as np, torch, imageio, cv2
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--tasks", default="castle:4321,jar:1234"); ap.add_argument("--chunks", default="6,10")
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
        if record: rec.append((i, round(tv), "high" if tv >= b1.boundary_t else "low", (lat - s * flow)[:, L:].clone()))
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
R = []; RP = os.path.join(a.out, "results.json")
if os.path.exists(RP): R = json.load(open(RP))
done = {(r["task"], r["chunk"], r["variant"]) for r in R}
for tk in a.tasks.split(","):
    short, seed = tk.split(":"); seed = int(seed); task = FULL[short]; T = [t for t in TASKS if t["id"] == task][0]
    V = [np.asarray(f) for f in imageio.get_reader(S + "/lingbot-world-v2/specwm_out/spec_v0/%s/base70_s%d/video.mp4" % (task, seed))]
    for ci in [int(x) for x in a.chunks.split(",")]:
        L = 3 * ci; F = 4 * (L + 3 - 1) + 1; c = b1.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=F, max_area=480 * 832); h, w = c["lat"][1:]
        fr = torch.from_numpy(np.stack([cv2.resize(f, (w * 8, h * 8), interpolation=cv2.INTER_AREA) for f in V[:F]])).float().div(127.5).sub(1).permute(3, 0, 1, 2).to(dev)
        with torch.no_grad(): pref = vae.encode([fr[:, :4 * (L - 1) + 1]])[0].float()
        gt = fr[:, 4 * (L - 1) + 1:]; fg, sg = feat(gt), sv(gt); lo, hi = 3, max(3, L - 12)
        print("[probe] %s s%d chunk %d: prefix %s, old latents %d..%d" % (short, seed, ci, tuple(pref.shape), lo, hi), flush=True)
        ref = None
        for vn in a.variants.split(","):
            zp = os.path.join(a.out, "z_%s_c%02d_%s.pt" % (short, ci, vn))
            if vn == "full_s0" and os.path.exists(zp):   # cached reference
                z = torch.load(zp).to(dev); px = dec(torch.cat([pref, z], 1))[:, 4 * (L - 1) + 1:]; ref = (feat(px), sv(px), z); continue
            if (task, ci, vn) in done and vn != "full_s0": continue
            keep = torch.ones(L, dtype=torch.bool); p = pref
            if vn == "blur2": p = blur_old(pref, 2, lo, hi)
            elif vn == "blur4": p = blur_old(pref, 4, lo, hi)
            elif vn == "drop_old": keep[lo:hi] = False
            elif vn.startswith("blur") and "_r" in vn:   # blur<k>_r<R>: everything between the sink chunk and the last R latents at 1/k resolution
                k, R_ = [int(x) for x in vn[4:].split("_r")]; p = blur_old(pref, k, 3, max(3, L - R_))
            t0 = time.time()
            if vn.startswith("sub") or vn.startswith("tsub"):   # sub<K>_r<R>: token subsample of the old part; tsub<K>_r<R>: every K-th old latent frame
                K_, R_ = [int(x) for x in vn[(3 if vn[0] == "s" else 4):].split("_r")]; z, rec = repaint_sub(c, pref, K_, max(4, L - R_), 0, vn[0] == "t"), []
            elif vn.startswith("mixL"):   # mixL<K>_r<R>: mixed-token forward + mixed-memory LoRA
                K_, R_ = [int(x) for x in vn[4:].split("_r")]; z, rec = repaint_mixed(c, pref, K_, max(4, L - R_), 0, lora=True), []
            elif vn.startswith("mix"):   # mix<K>_r<R>: real mixed-token forward
                K_, R_ = [int(x) for x in vn[3:].split("_r")]; z, rec = repaint_mixed(c, pref, K_, max(4, L - R_), 0), []
            elif vn == "nocam": z, rec = repaint(c, pref, keep, 0, nocam=True)   # camera conditioning removed (same noise as full_s0)
            elif vn.startswith("cfg"): z, rec = repaint(c, pref, keep, 0, cfg_steps=int(vn[3:]))   # cfg<K>: CFG only on the first K steps
            elif vn.startswith("reuse"): z, rec = repaint(c, pref, keep, 0, reuse=tuple(int(x) for x in vn[5:].split("-")))   # reuse<i-j-k>: skip those steps' forwards
            else: z, rec = repaint(c, p, keep, 1 if vn == "full_s1" else 0, record=(vn == "full_s0"))
            el = time.time() - t0
            px = dec(torch.cat([pref, z], 1))[:, 4 * (L - 1) + 1:]; f1, s1 = feat(px), sv(px)
            if vn == "full_s0": ref = (f1, s1, z); torch.save(z.cpu(), zp)
            r = dict(task=task, seed=seed, chunk=ci, variant=vn, sec=round(el, 1), dino_gt=dist(f1, s1, fg, sg)[0], col_gt=dist(f1, s1, fg, sg)[1])
            if ref is not None: r.update(dino_ref=dist(f1, s1, ref[0], ref[1])[0], col_ref=dist(f1, s1, ref[0], ref[1])[1], lat_ref=round(float((z - ref[2]).norm() / ref[2].norm()), 4))
            if rec:   # B: step commitment
                r["steps"] = []
                for i, tv, ex, x0 in rec:
                    pxi = dec(torch.cat([pref, x0], 1))[:, 4 * (L - 1) + 1:]; fi, si = feat(pxi), sv(pxi)
                    r["steps"].append(dict(i=i, t=tv, expert=ex, dino_final=dist(fi, si, f1, s1)[0], col_final=dist(fi, si, f1, s1)[1], dino_gt=dist(fi, si, fg, sg)[0]))
                    cv2.imwrite(os.path.join(a.out, "%s_c%02d_step%02d.jpg" % (short, ci, i)), cv2.cvtColor(((pxi[:, 6].permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(a.out, "%s_c%02d_%s.jpg" % (short, ci, vn)), cv2.cvtColor(((px[:, 6].permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(a.out, "%s_c%02d_gt.jpg" % (short, ci)), cv2.cvtColor(((gt[:, 6].permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            R = [x for x in R if (x["task"], x["chunk"], x["variant"]) != (task, ci, vn)] + [r]; json.dump(R, open(RP, "w"), indent=1)
            print("[probe]", {k: v for k, v in r.items() if k != "steps"}, flush=True)
print("[probe] done", flush=True)
