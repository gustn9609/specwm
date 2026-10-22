"""On-policy teacher data for the Fast LoRA (DAgger step): from a FAST rollout of a task, at chunk boundaries c in --chunks,
Base generates a long continuation (I2V from the rollout's last prefix frame, poses from that time, --frames frames, --steps steps, CFG 5)
and its first chunk (the 3 latents after the image latent) is the target for chunk c given Fast's own prefix.
Saves <out>/<task>_s<seed>_c<c>.pt = dict(prefix=[16,3c,h,w] bf16 (VAE-encoded Fast frames), target=[16,3,h,w], task, seed, chunk).
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.onpolicy_gen --out DIR --tasks a,b,c [--chunks 3,6,9 --frames 81 --steps 40]"""
import argparse, os, sys, json, time, numpy as np, torch, imageio, cv2
S = "/home/hk28965/specwm"; V0 = S + "/lingbot-world-v2/specwm_out/spec_v0"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--tasks", required=True); ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--chunks", default="3,6,9"); ap.add_argument("--frames", type=int, default=81); ap.add_argument("--steps", type=int, default=40); ap.add_argument("--shift", type=float, default=10.0); ap.add_argument("--guide", type=float, default=5.0)
ap.add_argument("--rollout", default="never", help="run name of the Fast rollout to continue from (<task>/<rollout>_s<seed>/video.mp4)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world"); os.chdir(S + "/lingbot-world")
import importlib, importlib.util
spec = importlib.util.spec_from_file_location("specwm_v1", S + "/lingbot-world/specwm/__init__.py", submodule_search_locations=[S + "/lingbot-world/specwm"]); pkg = importlib.util.module_from_spec(spec); sys.modules["specwm_v1"] = pkg; spec.loader.exec_module(pkg)
V1Binding = importlib.import_module("specwm_v1.binding_v1").V1Binding
dev = torch.device("cuda:0"); b1 = V1Binding(S + "/lingbot-world/lingbot-world-base-cam", t5_cpu=True, device=str(dev))
# both experts resident on the GPU (an 81-frame window fits next to them); NO pinned host copies — six processes pinning
# 2 x 28 GB each exhausted the host's 755 GB and the kernel OOM-killed everything (incl. the Claude session) twice on 9/25
EXP = {"high": b1.pipe.high_noise_model.to(dev), "low": b1.pipe.low_noise_model.to(dev)}; b1.device = dev
def expert(name): return EXP[name]
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
vae = b1.pipe.vae; T = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; CS = 3
@torch.no_grad()
def generate(c, seed):
    """plain I2V sampling of the whole window (no prefix): --steps UniPC, CFG, two experts"""
    sch = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False); sch.set_timesteps(a.steps, device=dev, shift=a.shift)
    g = torch.Generator(device=dev).manual_seed(seed); lat = torch.randn((16,) + tuple(c["lat"]), generator=g, device=dev)
    for i, t in enumerate(sch.timesteps):
        tv = float(t); m = expert("high" if tv >= b1.boundary_t else "low"); tt = torch.full((1,), tv, device=dev)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            fc = m([lat.to(torch.bfloat16)], t=tt, context=[c["ctx"][0]], seq_len=c["seq_len"], y=[c["y"]], dit_cond_dict=c["dit_cond"])[0].float()
            fu = m([lat.to(torch.bfloat16)], t=tt, context=c["ctxn"], seq_len=c["seq_len"], y=[c["y"]], dit_cond_dict=c["dit_cond"])[0].float()
        lat = sch.step((fu + a.guide * (fc - fu)).unsqueeze(0), t, lat.unsqueeze(0), return_dict=False, generator=g)[0].squeeze(0)
    return lat
for task in a.tasks.split(","):
    vp = os.path.join(V0, task, "%s_s%d" % (a.rollout, a.seed), "video.mp4")
    if not os.path.exists(vp): print("[onpolicy] missing rollout", vp, flush=True); continue
    t = T[task]; c0 = b1.prepare(t["image"], t.get("camera_WM", ""), t["action"], frame_num=161, max_area=480 * 832); h, w = c0["lat"][1:]
    V = [np.asarray(f) for f in imageio.get_reader(vp)]
    fr = torch.from_numpy(np.stack([cv2.resize(f, (w * 8, h * 8), interpolation=cv2.INTER_AREA) for f in V])).float().div(127.5).sub(1).permute(3, 0, 1, 2).to(dev)
    for ci in [int(x) for x in a.chunks.split(",")]:
        op = os.path.join(a.out, "%s_s%d_c%d.pt" % (task, a.seed, ci))
        if os.path.exists(op): continue
        L = CS * ci; npre = 4 * (L - 1) + 1   # frames covered by the prefix latents
        with torch.no_grad(): pref = vae.encode([fr[:, :npre]])[0].float()
        img = fr[:, npre - 1].clamp(-1, 1)   # the last prefix frame, as the I2V image
        t0 = time.time(); c = b1.prepare(img, t.get("camera_WM", ""), t["action"], frame_num=a.frames, max_area=480 * 832, pose_offset=npre - 1)
        lat = generate(c, a.seed * 13 + ci)
        target = lat[:, 1:1 + CS]   # latent 0 = the image; the next 3 latents = 12 frames = chunk ci
        torch.save(dict(prefix=pref.to(torch.bfloat16).cpu(), target=target.to(torch.bfloat16).cpu(), task=task, seed=a.seed, chunk=ci, frames=a.frames, steps=a.steps), op)
        print("[onpolicy] %s c%d: prefix %s target %s  %.0fs" % (task, ci, tuple(pref.shape), tuple(target.shape), time.time() - t0), flush=True)
print("[onpolicy] done", flush=True)
