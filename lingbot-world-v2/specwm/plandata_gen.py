"""Training data for plan -> render: for each training video (name from mixlora/cache), Base makes (1) the low-res whole-clip PLAN
(576x304, --plan-steps UniPC, CFG on --cfg-steps, flow reuse at --reuse) exactly like the test-time plan, and (2) a plan-conditioned
480p TARGET: the plan upscaled -> VAE -> noised to sigma --s-start -> Base (both experts, CFG) denoises the remaining --refine-steps schedule.
Writes <out>/cache/<name>.pt = dict(lat=target [16,41,h,w] bf16, scene) and <out>/hints/<name>.pt = dict(hint_pix = VAE(upscaled plan) (the
test-time hint), hint_lo = plan latents, size_lo), plus <out>/video/<name>_{plan,target}.mp4 and a json with timings.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.plandata_gen --out DIR --names all|a,b --shard k/n [--s-start 0.7 --refine-steps 30]"""
import argparse, os, sys, json, time, glob, numpy as np, torch, imageio, cv2
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--names", default="all"); ap.add_argument("--shard", default="0/1")
ap.add_argument("--cache", default=S + "/lingbot-world-v2/specwm_out/mixlora/cache")
ap.add_argument("--area", type=int, default=184320); ap.add_argument("--plan-steps", type=int, default=16); ap.add_argument("--cfg-steps", type=int, default=6); ap.add_argument("--reuse", default="9,11,13")
ap.add_argument("--refine-steps", type=int, default=30); ap.add_argument("--s-start", type=float, default=0.7); ap.add_argument("--refine-cfg-steps", type=int, default=-1)
ap.add_argument("--shift", type=float, default=10.0); ap.add_argument("--guide", type=float, default=5.0); ap.add_argument("--tag", default="", help="suffix for the output names (sigma sweeps)")
ap.add_argument("--seed-override", type=int, default=-1); ap.add_argument("--spec", default="", help="JSON list of {name, scene, seed, action?}: explicit clips (scene = task id or S-scene code; action overrides the task's camera path)")
a = ap.parse_args(); a.out = os.path.abspath(a.out); a.cache = os.path.abspath(a.cache)
for d in ("cache", "hints", "video"): os.makedirs(os.path.join(a.out, d), exist_ok=True)
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world"); os.chdir(S + "/lingbot-world")
import importlib, importlib.util
spec = importlib.util.spec_from_file_location("specwm_v1", S + "/lingbot-world/specwm/__init__.py", submodule_search_locations=[S + "/lingbot-world/specwm"]); pkg = importlib.util.module_from_spec(spec); sys.modules["specwm_v1"] = pkg; spec.loader.exec_module(pkg)
V1Binding = importlib.import_module("specwm_v1.binding_v1").V1Binding
dev = torch.device("cuda:0"); b1 = V1Binding(S + "/lingbot-world/lingbot-world-base-cam", t5_cpu=True, device=str(dev))
EXP = {"high": b1.pipe.high_noise_model.to(dev), "low": b1.pipe.low_noise_model.to(dev)}; b1.device = dev   # both experts resident, nothing pinned on the host
LOC = {"high": "cuda", "low": "cuda"}
def expert(name):
    """the named expert on the GPU; at 480p both experts do not fit next to the activations, so the other one is parked on the host (plain copy, no pinning)"""
    for k, m in EXP.items():
        want = "cuda" if k == name else "cpu"
        if LOC[k] != want: m.to(dev if want == "cuda" else "cpu"); LOC[k] = want; torch.cuda.empty_cache()
    return EXP[name]
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
vae = b1.pipe.vae
SC = {sc: dict(img=S + "/lingbot-world-v2/examples/%s/image.jpg" % sc, prompt=open(S + "/p_%s.txt" % sc).read().strip(), act=S + "/rt2_action_%s" % sc) for sc in ("00", "01", "02", "05")}
TASKS = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
def cond_of(scene):
    if scene in SC: return SC[scene]["img"], SC[scene]["prompt"], SC[scene]["act"]
    t = TASKS[scene]; return t["image"], t.get("camera_WM", ""), t["action"]
def parse(name):
    """cache name -> (scene, seed): S05_s1234 -> ("05", 1234); alka_seltzer_fizz_base70_s1234 -> ("alka_seltzer_fizz", 1234)"""
    seed = int(name.rsplit("_s", 1)[1]); head = name.rsplit("_s", 1)[0]
    return (head[1:3], seed) if head.startswith("S") and len(head) == 3 else (head.replace("_base70", ""), seed)
@torch.no_grad()
def sample(c, seed, steps, cfg_steps, reuse, lat0=None, s_start=1.0):
    sch = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False); sch.set_timesteps(steps, device=dev, shift=a.shift)
    g = torch.Generator(device=dev).manual_seed(seed); noise = torch.randn((16,) + tuple(c["lat"]), generator=g, device=dev); flow = None; i0 = 0; lat = noise
    if lat0 is not None:
        i0 = next(i for i in range(len(sch.timesteps)) if float(sch.sigmas[i]) <= s_start + 1e-4); s = float(sch.sigmas[i0]); lat = (1 - s) * lat0 + s * noise
    for i, t in enumerate(sch.timesteps):
        if i < i0: continue
        tv = float(t); m = expert("high" if tv >= b1.boundary_t else "low"); tt = torch.full((1,), tv, device=dev)
        if not (i in reuse and flow is not None):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                fc = m([lat.to(torch.bfloat16)], t=tt, context=[c["ctx"][0]], seq_len=c["seq_len"], y=[c["y"]], dit_cond_dict=c["dit_cond"])[0].float()
                if cfg_steps < 0 or (i - i0) < cfg_steps:
                    fu = m([lat.to(torch.bfloat16)], t=tt, context=c["ctxn"], seq_len=c["seq_len"], y=[c["y"]], dit_cond_dict=c["dit_cond"])[0].float(); flow = fu + a.guide * (fc - fu)
                else: flow = fc
        lat = sch.step(flow.unsqueeze(0), t, lat.unsqueeze(0), return_dict=False, generator=g)[0].squeeze(0)
    return lat, i0
def to_video(lat, path):
    with torch.no_grad(): px = vae.decode([lat])[0].clamp(-1, 1)
    v = ((px.permute(1, 2, 3, 0).float().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    imageio.mimwrite(path, list(v), fps=16, codec="libx264", quality=8); return v
if a.spec:
    ITEMS = [(d["name"], d["scene"], int(d["seed"]), d.get("action")) for d in json.load(open(a.spec))]
else:
    names = sorted(os.path.basename(f)[:-3] for f in glob.glob(a.cache + "/*.pt")) if a.names == "all" else a.names.split(",")
    ITEMS = [(nm,) + parse(nm) + (None,) for nm in names]
k, n = [int(x) for x in a.shard.split("/")]; reuse = {int(x) for x in a.reuse.split(",") if x}
for idx, (name, scene, seed, act_over) in enumerate(ITEMS):
    if idx % n != k: continue
    oname = name + a.tag; op = os.path.join(a.out, "cache", oname + ".pt")
    if os.path.exists(op): continue
    seed = a.seed_override if a.seed_override >= 0 else seed; img, prompt, act = cond_of(scene); act = act_over or act; T = {}
    t0 = time.time(); c_lo = b1.prepare(img, prompt, act, frame_num=161, max_area=a.area)
    plan_lat, _ = sample(c_lo, seed, a.plan_steps, a.cfg_steps, reuse); torch.cuda.synchronize(); T["plan_s"] = round(time.time() - t0, 1)
    pv = to_video(plan_lat, os.path.join(a.out, "video", oname + "_plan.mp4"))
    t0 = time.time(); c_hi = b1.prepare(img, prompt, act, frame_num=161, max_area=480 * 832); h, w = c_hi["lat"][1:]
    up = torch.from_numpy(np.stack([cv2.resize(f, (w * 8, h * 8), interpolation=cv2.INTER_CUBIC) for f in pv])).float().div(127.5).sub(1).permute(3, 0, 1, 2).to(dev)
    with torch.no_grad(): lat0 = vae.encode([up])[0].float()
    del up
    tgt, i0 = sample(c_hi, seed * 31 + 7, a.refine_steps, a.refine_cfg_steps, set(), lat0=lat0, s_start=a.s_start); torch.cuda.synchronize(); T["refine_s"] = round(time.time() - t0, 1); T["refine_from_step"] = i0
    to_video(tgt, os.path.join(a.out, "video", oname + "_target.mp4"))
    torch.save(dict(lat=tgt.to(torch.bfloat16).cpu(), scene=scene, seed=seed, s_start=a.s_start, action=act), op)
    torch.save(dict(hint_pix=lat0.to(torch.bfloat16).cpu(), hint_lo=plan_lat.to(torch.bfloat16).cpu(), size_lo=tuple(plan_lat.shape[2:])), os.path.join(a.out, "hints", oname + ".pt"))
    json.dump(dict(name=oname, scene=scene, seed=seed, area=a.area, plan_steps=a.plan_steps, cfg_steps=a.cfg_steps, reuse=a.reuse, refine_steps=a.refine_steps, s_start=a.s_start, **T), open(os.path.join(a.out, "video", oname + ".json"), "w"), indent=1)
    print("[plandata] %s: plan %.0f s, refine %.0f s (from step %d of %d), target %s" % (oname, T["plan_s"], T["refine_s"], i0, a.refine_steps, tuple(tgt.shape)), flush=True)
    del tgt, lat0, plan_lat; torch.cuda.empty_cache()
print("[plandata] done", flush=True)
