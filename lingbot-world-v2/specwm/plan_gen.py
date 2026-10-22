"""Low-res whole-clip Base PLAN for plan -> render: one Base generation of the full clip at --area (default 576x304), --steps UniPC,
CFG on the first --cfg-steps steps, flow reuse at --reuse. Saves video.mp4, plan latents (plan_lat.pt) and timing (plan.json).
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.plan_gen --task T --seed S --out DIR [--cfg-steps 6 --reuse 9,11,13]
(original docstring follows) On-policy teacher data for the Fast LoRA (DAgger step): from a FAST rollout of a task, at chunk boundaries c in --chunks,
Base generates a long continuation (I2V from the rollout's last prefix frame, poses from that time, --frames frames, --steps steps, CFG 5)
and its first chunk (the 3 latents after the image latent) is the target for chunk c given Fast's own prefix.
Saves <out>/<task>_s<seed>_c<c>.pt = dict(prefix=[16,3c,h,w] bf16 (VAE-encoded Fast frames), target=[16,3,h,w], task, seed, chunk).
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.onpolicy_gen --out DIR --tasks a,b,c [--chunks 3,6,9 --frames 81 --steps 40]"""
import argparse, os, sys, json, time, numpy as np, torch, imageio, cv2
S = "/home/hk28965/specwm"; V0 = S + "/lingbot-world-v2/specwm_out/spec_v0"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--task", required=True); ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--frames", type=int, default=161); ap.add_argument("--steps", type=int, default=16); ap.add_argument("--shift", type=float, default=10.0); ap.add_argument("--guide", type=float, default=5.0)
ap.add_argument("--area", type=int, default=184320); ap.add_argument("--cfg-steps", type=int, default=-1, help="CFG only on the first K steps"); ap.add_argument("--reuse", default="", help="comma list of step indices whose forward is skipped (previous flow reused)")
ap.add_argument("--lora", default="", help="merge a public Wan2.2 I2V A14B few-step LoRA into both experts: 1022 (lightx2v 4-step) | seko (Lightning Seko-V1); use with --steps 4 --cfg-steps 0 --shift 5"); ap.add_argument("--lora-scale", type=float, default=1.0); ap.add_argument("--window", type=int, default=0, help="streaming plan: plan --window frames at a time (e.g. 81); each next window is I2V from the previous window's last low-res frame with the matching pose offset"); ap.add_argument("--planlora", default="", help="plan_lora.pt from planlora_train.py: merged on top (per expert)")
ap.add_argument("--offload", type=int, default=0, help="[10/15] experts live on the CPU; the needed one is moved to the GPU on demand (bf16 at 832x480x161 does not fit with both resident)"); ap.add_argument("--fp8", type=int, default=0, help="[10/15 bid speed] swap every Linear of both experts to FP8 (specwm/fast_patch.fp8_swap of the v2 code)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world"); os.chdir(S + "/lingbot-world")
import importlib, importlib.util
spec = importlib.util.spec_from_file_location("specwm_v1", S + "/lingbot-world/specwm/__init__.py", submodule_search_locations=[S + "/lingbot-world/specwm"]); pkg = importlib.util.module_from_spec(spec); sys.modules["specwm_v1"] = pkg; spec.loader.exec_module(pkg)
V1Binding = importlib.import_module("specwm_v1.binding_v1").V1Binding
dev = torch.device("cuda:0"); b1 = V1Binding(os.environ.get("SPECWM_BASE_CKPT") or (S + "/lingbot-world/lingbot-world-base-cam"), t5_cpu=True, device=str(dev))
# both experts resident on the GPU (an 81-frame window fits next to them); NO pinned host copies — six processes pinning
# 2 x 28 GB each exhausted the host's 755 GB and the kernel OOM-killed everything (incl. the Claude session) twice on 9/25
EXP = {"high": b1.pipe.high_noise_model.to(dev if not a.offload else "cpu"), "low": b1.pipe.low_noise_model.to(dev if not a.offload else "cpu")}; b1.device = dev
TMX = dict(fwd=0.0, swap=0.0, n_fwd=0, n_swap=0); _RES = [None]
def expert(name):
    if not a.offload: return EXP[name]
    if _RES[0] != name:
        _t = time.time()
        if _RES[0] is not None: EXP[_RES[0]].cpu(); torch.cuda.empty_cache()
        EXP[name].to(dev); torch.cuda.synchronize(); TMX["swap"] += time.time() - _t; TMX["n_swap"] += 1; _RES[0] = name
    return EXP[name]
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
LORA = {"1022": ("/home/hk28965/specwm/hf/lightx2v/Wan2.2-Distill-Loras/wan2.2_i2v_A14b_%s_noise_lora_rank64_lightx2v_4step_1022.safetensors", False),
        "seko": ("/home/hk28965/specwm/hf/lightx2v/Wan2.2-Lightning/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/%s_noise_model.safetensors", True)}
def merge_lora(Wd, path, use_alpha, scale):
    from safetensors import safe_open
    n_lin = n_b = n_d = 0
    with safe_open(path, "pt") as fh:
        keys = set(fh.keys())
        for k in keys:
            if not k.endswith(".lora_down.weight"): continue
            base = k[len("diffusion_model."):-len(".lora_down.weight")]; wk = base + ".weight"
            if wk not in Wd: continue
            dn = fh.get_tensor(k).float(); up = fh.get_tensor("diffusion_model." + base + ".lora_up.weight").float()
            sc = scale * (float(fh.get_tensor("diffusion_model." + base + ".alpha")) / dn.shape[0] if use_alpha and ("diffusion_model." + base + ".alpha") in keys else 1.0)
            dW = (up.flatten(1) @ dn.flatten(1)).reshape(Wd[wk].shape).to(Wd[wk].device) * sc; Wd[wk].copy_((Wd[wk].float() + dW).to(Wd[wk].dtype)); n_lin += 1
        for k in keys:
            if k.endswith(".diff_b") or k.endswith(".diff"):
                base = k[len("diffusion_model."):].rsplit(".", 1)[0]; wk = base + (".bias" if k.endswith(".diff_b") else ".weight")
                if wk in Wd and Wd[wk].shape == fh.get_tensor(k).shape:
                    Wd[wk].copy_((Wd[wk].float() + scale * fh.get_tensor(k).float().to(Wd[wk].device)).to(Wd[wk].dtype)); n_b += k.endswith(".diff_b"); n_d += k.endswith(".diff")
    print("[lora] %s: merged %d low-rank, %d bias, %d weight deltas" % (os.path.basename(path), n_lin, n_b, n_d), flush=True)
if a.lora:
    pth, ua = LORA[a.lora]
    with torch.no_grad():
        for ex in ("high", "low"): merge_lora(dict(EXP[ex].named_parameters()), pth % ex, ua, a.lora_scale)
if a.planlora:
    _st = torch.load(a.planlora)
    with torch.no_grad():
        for ex, lst in _st["A"].items():
            mods = []
            for blk in EXP[ex].blocks:
                mods += [blk.self_attn.q, blk.self_attn.k, blk.self_attn.v, blk.self_attn.o, blk.ffn[0], blk.ffn[2]]
            for mod, A_, B_ in zip(mods, lst, _st["B"][ex]): mod.weight.add_((B_.to(dev).float() @ A_.to(dev).float()).to(mod.weight.dtype))
            print("[planlora] merged %d modules into the %s expert (step %s)" % (len(mods), ex, _st.get("step")), flush=True)
if a.fp8:
    _fs = importlib.util.spec_from_file_location("fast_patch_v2", S + "/lingbot-world-v2/specwm/fast_patch.py"); _fp = importlib.util.module_from_spec(_fs); _fs.loader.exec_module(_fp)
    for ex in ("high", "low"): _fp.fp8_swap(EXP[ex], True, keep=False)
    torch.cuda.synchronize(); print("[fp8] experts swapped: %.1f GiB allocated" % (torch.cuda.memory_allocated() / 2**30), flush=True)
vae = b1.pipe.vae; T = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; CS = 3
@torch.no_grad()
def generate(c, seed):
    """plain I2V sampling of the whole window (no prefix): --steps UniPC, CFG, two experts"""
    sch = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False); sch.set_timesteps(a.steps, device=dev, shift=a.shift)
    g = torch.Generator(device=dev).manual_seed(seed); lat = torch.randn((16,) + tuple(c["lat"]), generator=g, device=dev); flow = None
    reuse = {int(x) for x in a.reuse.split(",") if x}
    for i, t in enumerate(sch.timesteps):
        tv = float(t); m = expert("high" if tv >= b1.boundary_t else "low"); tt = torch.full((1,), tv, device=dev)
        if not (i in reuse and flow is not None):
            torch.cuda.synchronize(); _tf = time.time()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                fc = m([lat.to(torch.bfloat16)], t=tt, context=[c["ctx"][0]], seq_len=c["seq_len"], y=[c["y"]], dit_cond_dict=c["dit_cond"])[0].float()
                if a.cfg_steps < 0 or i < a.cfg_steps:
                    fu = m([lat.to(torch.bfloat16)], t=tt, context=c["ctxn"], seq_len=c["seq_len"], y=[c["y"]], dit_cond_dict=c["dit_cond"])[0].float(); flow = fu + a.guide * (fc - fu)
                else: flow = fc
            torch.cuda.synchronize(); TMX["fwd"] += time.time() - _tf; TMX["n_fwd"] += 1 + (a.cfg_steps < 0 or i < a.cfg_steps)
        lat = sch.step(flow.unsqueeze(0), t, lat.unsqueeze(0), return_dict=False, generator=g)[0].squeeze(0)
    return lat

t = T[a.task]; t0 = time.time(); TW = []
if a.window <= 0:
    c = b1.prepare(t["image"], t.get("camera_WM", ""), t["action"], frame_num=a.frames, max_area=a.area)
    lat = generate(c, a.seed); torch.cuda.synchronize(); gen_s = time.time() - t0
else:   # windowed / streaming plan: window k covers frames [f0, f0 + window); the next window starts from this window's last frame
    lats = []; f0 = 0; img = t["image"]; k = 0
    while f0 < a.frames - 1:   # windows overlap by one frame (the conditioning image); stop once the last frame is covered
        F = min(a.window, a.frames - f0); F = max(5, 4 * ((F - 1) // 4) + 1); tw = time.time()
        c = b1.prepare(img, t.get("camera_WM", ""), t["action"], frame_num=F, max_area=a.area, pose_offset=f0)
        lw = generate(c, a.seed * 17 + k); torch.cuda.synchronize(); TW.append(round(time.time() - tw, 1)); print("[plan] window %d frames %d-%d: %.0f s (experts on %s)" % (k, f0, f0 + F - 1, TW[-1], {kk: str(next(m.parameters()).device) for kk, m in EXP.items()}), flush=True)
        lats.append(lw if k == 0 else lw[:, 1:])   # the next window's first latent is the conditioning image (= previous last frame): drop it
        with torch.no_grad(): img = vae.decode([lw[:, -1:]])[0][:, -1].clamp(-1, 1).float()   # last decoded frame of this window (low res)
        f0 += F - 1; k += 1
    lat = torch.cat(lats, 1)[:, :(a.frames - 1) // 4 + 1]; gen_s = time.time() - t0
with torch.no_grad(): px = vae.decode([lat])[0].clamp(-1, 1)
v = ((px.permute(1, 2, 3, 0).float().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
imageio.mimwrite(os.path.join(a.out, "video.mp4"), list(v), fps=16, codec="libx264", quality=8)
torch.save(dict(lat=lat.to(torch.bfloat16).cpu(), task=a.task, seed=a.seed, area=a.area, steps=a.steps, cfg_steps=a.cfg_steps, reuse=a.reuse), os.path.join(a.out, "plan_lat.pt"))
json.dump(dict(task=a.task, seed=a.seed, area=a.area, steps=a.steps, cfg_steps=a.cfg_steps, reuse=a.reuse, gen_s=round(gen_s, 1), fwd_s=round(TMX["fwd"], 1), n_fwd=TMX["n_fwd"], swap_s=round(TMX["swap"], 1), n_swap=TMX["n_swap"], fp8=a.fp8, offload=a.offload, lat=list(lat.shape), window=a.window, window_s=TW), open(os.path.join(a.out, "plan.json"), "w"), indent=1)
print("[plan] %s s%d area %d steps %d cfg %d reuse %s window %d: %.0f s %s, latents %s" % (a.task, a.seed, a.area, a.steps, a.cfg_steps, a.reuse, a.window, gen_s, TW, tuple(lat.shape)), flush=True)
