"""On-policy targets for the texture LoRA (DAgger with the plan): for chunk c of a Fast plan-rollout (plan_rollout.py), Base regenerates
chunk c given FAST'S OWN prefix (RePaint: the last --ctx chunks of the prefix are re-noised and replaced every step) starting from the plan
hint chunk at sigma --s-start (low-noise expert, CFG, 30-step shift-10 schedule). The target is continuous with Fast's prefix and follows the plan.
Saves <out>/<name>_c<c>.pt = dict(prefix=[16,3c,h,w] Fast latents, target=[16,3,h,w], hint=[16,3,h,w], z=[4,16,3,h,w] node inputs of the rollout,
x0n=[4,...] the rollout's node outputs, chunk, scene, seed).
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.onpolicy2_gen --out DIR [--chunks 2,5,8,11 --ctx 5 --s-start 0.9 --shard k/n]"""
import argparse, os, sys, glob, json, time, numpy as np, torch
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--rollout", default=S + "/lingbot-world-v2/specwm_out/onpol2/rollout"); ap.add_argument("--data", default=S + "/lingbot-world-v2/specwm_out/plandata")
ap.add_argument("--chunks", default="2,5,8,11"); ap.add_argument("--ctx", type=int, default=5, help="prefix chunks in the Base window"); ap.add_argument("--s-start", type=float, default=0.9); ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--shift", type=float, default=10.0); ap.add_argument("--guide", type=float, default=5.0); ap.add_argument("--shard", default="0/1"); ap.add_argument("--names", default="all"); ap.add_argument("--init-node", type=int, default=-1, help="DAgger labelling: start Base from the rollout's actual state entering this Fast node (z[k], sigma_k = 1.00/0.94/0.83/0.63) instead of the plan hint at --s-start")
a = ap.parse_args(); a.out = os.path.abspath(a.out); a.rollout = os.path.abspath(a.rollout); a.data = os.path.abspath(a.data); os.makedirs(a.out, exist_ok=True)
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world"); os.chdir(S + "/lingbot-world")
import importlib, importlib.util
spec = importlib.util.spec_from_file_location("specwm_v1", S + "/lingbot-world/specwm/__init__.py", submodule_search_locations=[S + "/lingbot-world/specwm"]); pkg = importlib.util.module_from_spec(spec); sys.modules["specwm_v1"] = pkg; spec.loader.exec_module(pkg)
V1Binding = importlib.import_module("specwm_v1.binding_v1").V1Binding
dev = torch.device("cuda:0"); b1 = V1Binding(S + "/lingbot-world/lingbot-world-base-cam", t5_cpu=True, device=str(dev))
LOW = b1.pipe.low_noise_model.to(dev); b1.pipe.high_noise_model.to("cpu"); b1.device = dev; torch.cuda.empty_cache()   # only the low-noise expert is needed (sigma <= 0.9)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
vae = b1.pipe.vae; CS = 3
SC = {sc: dict(img=S + "/lingbot-world-v2/examples/%s/image.jpg" % sc, prompt=open(S + "/p_%s.txt" % sc).read().strip(), act=S + "/rt2_action_%s" % sc) for sc in ("00", "01", "02", "05")}
TASKS = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
def cond_of(scene):
    if scene in SC: return SC[scene]["img"], SC[scene]["prompt"], SC[scene]["act"]
    t = TASKS[scene]; return t["image"], t.get("camera_WM", ""), t["action"]
def flow_at(lat, tv, c):
    tt = torch.full((1,), tv, device=dev)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        fc = LOW([lat.to(torch.bfloat16)], t=tt, context=[c["ctx"][0]], seq_len=c["seq_len"], y=[c["y"]], dit_cond_dict=c["dit_cond"])[0].float()
        fu = LOW([lat.to(torch.bfloat16)], t=tt, context=c["ctxn"], seq_len=c["seq_len"], y=[c["y"]], dit_cond_dict=c["dit_cond"])[0].float()
    return fu + a.guide * (fc - fu)
names = sorted(os.path.basename(f)[:-3] for f in glob.glob(a.rollout + "/*.pt")) if a.names == "all" else a.names.split(",")
k, n = [int(x) for x in a.shard.split("/")]; CH = [int(x) for x in a.chunks.split(",")]
for idx, name in enumerate(names):
    if idx % n != k: continue
    if all(os.path.exists(os.path.join(a.out, "%s_c%d.pt" % (name, c))) for c in CH): continue
    R = torch.load(os.path.join(a.rollout, name + ".pt")); D = torch.load(os.path.join(a.data, "cache", name + ".pt")); H = torch.load(os.path.join(a.data, "hints", name + ".pt"))
    lat_all = R["lat"].to(dev).float(); hint_all = H["hint_pix"].to(dev).float(); img, prompt, act = cond_of(D["scene"]); seed = int(R["seed"])
    for c in CH:
        op = os.path.join(a.out, "%s_c%d.pt" % (name, c))
        if os.path.exists(op): continue
        t0 = time.time(); j = max(0, c - a.ctx) * CS; pref = lat_all[:, j:c * CS]; L = pref.shape[1]; F = 4 * (L + CS - 1) + 1
        if j == 0: cnd = b1.prepare(img, prompt, act, frame_num=F, max_area=480 * 832)
        else:
            with torch.no_grad(): im0 = vae.decode([lat_all[:, j:j + 1]])[0][:, 0].clamp(-1, 1).float()
            cnd = b1.prepare(im0, prompt, act, frame_num=F, max_area=480 * 832, pose_offset=4 * j)
        assert tuple(cnd["lat"][1:]) == tuple(pref.shape[2:]), (cnd["lat"], pref.shape)
        sch = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False); sch.set_timesteps(a.steps, device=dev, shift=a.shift)
        g = torch.Generator(device=dev).manual_seed(seed * 7919 + c); hint = hint_all[:, c * CS:(c + 1) * CS]
        if a.init_node >= 0:   # same state as Fast's node k: Base denoises the rollout's z[k] from sigma_k (the expert's answer for the learner's exact input)
            SIGK = [1.0, 0.94, 0.83, 0.63][a.init_node]; i0 = next(i for i in range(len(sch.timesteps)) if float(sch.sigmas[i]) <= SIGK + 1e-4); s0 = float(sch.sigmas[i0])
            zk = R["z"][c][a.init_node].to(dev).float(); x0k = R["x0n"][c][max(0, a.init_node - 1)].to(dev).float()
            lat = torch.cat([pref, zk if abs(s0 - SIGK) < 1e-3 else (1 - s0) * x0k + s0 * torch.randn(hint.shape, generator=g, device=dev)], 1)
        else:
            i0 = next(i for i in range(len(sch.timesteps)) if float(sch.sigmas[i]) <= a.s_start + 1e-4); s0 = float(sch.sigmas[i0])
            lat = torch.cat([pref, (1 - s0) * hint + s0 * torch.randn(hint.shape, generator=g, device=dev)], 1)
        z_init = lat[:, L:].clone()   # the exact input the learner is trained on (deterministic pairing: same noise as the target's init)
        for i, t in enumerate(sch.timesteps):
            if i < i0: continue
            s = float(sch.sigmas[i]); assert float(t) < b1.boundary_t
            lat[:, :L] = (1 - s) * pref + s * torch.randn(pref.shape, generator=g, device=dev)   # RePaint: the prefix is re-noised every step
            flow = flow_at(lat, float(t), cnd)
            lat = sch.step(flow.unsqueeze(0), t, lat.unsqueeze(0), return_dict=False, generator=g)[0].squeeze(0)
        tgt = lat[:, L:L + CS].float()
        torch.save(dict(prefix=lat_all[:, :c * CS].to(torch.bfloat16).cpu(), target=tgt.to(torch.bfloat16).cpu(), hint=hint.to(torch.bfloat16).cpu(), z=R["z"][c], x0n=R["x0n"][c], chunk=c, scene=D["scene"], seed=seed, s_start=a.s_start, init_node=a.init_node, s0=s0, z_init=z_init.to(torch.bfloat16).cpu()), op)
        print("[onpol2] %s c%d: window %d+%d latents, %d steps, %.0f s" % (name, c, L, CS, a.steps - i0, time.time() - t0), flush=True)
print("[onpol2] done", flush=True)
