"""standalone: KV RePaint of chunk 1 given a clean chunk 0 (VAE-encoded from the Base 70-step video) vs full RePaint. Saves decoded frames."""
import os, sys, json, torch, numpy as np, imageio, cv2
S = "/home/hk28965/specwm"; sys.path.insert(0, S + "/lingbot-world"); os.chdir(S + "/lingbot-world")
os.environ["SPECWM_NO_FLASH"] = "1"
import importlib, importlib.util
spec = importlib.util.spec_from_file_location("specwm_v1", S + "/lingbot-world/specwm/__init__.py", submodule_search_locations=[S + "/lingbot-world/specwm"]); pkg = importlib.util.module_from_spec(spec); sys.modules["specwm_v1"] = pkg; spec.loader.exec_module(pkg)
V1Binding = importlib.import_module("specwm_v1.binding_v1").V1Binding
dev = torch.device("cuda:0"); b1 = V1Binding(S + "/lingbot-world/lingbot-world-base-cam", t5_cpu=True, device=str(dev))
M = {"high": b1.pipe.high_noise_model.to(dev), "low": b1.pipe.low_noise_model.to(dev)}; b1.device = dev
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
T = [t for t in json.load(open(S + "/data/stevo/tasks_camera.json")) if t["id"] == "sand_filling_up_jar"][0]
c = b1.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=21, max_area=480 * 832); lat = c["lat"]; H, W = lat[1] * 8, lat[2] * 8
fr = [np.asarray(f) for f in imageio.get_reader(S + "/lingbot-world-v2/specwm_out/spec_v0/sand_filling_up_jar/base70_s1234/video.mp4")][:9]
v = torch.from_numpy(np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in fr])).float().div(127.5).sub(1).permute(3, 0, 1, 2).to(dev)
vae = b1.pipe.vae
with torch.no_grad(): pref = vae.encode([v])[0].float()
print("pref", tuple(pref.shape), "lat", lat, flush=True)
sys.path.insert(0, S + "/lingbot-world-v2"); from specwm.base_cache import BaseKV
cam = c["dit_cond"]["c2ws_plucker_emb"][0] if c["dit_cond"] else None; guide = 5.0; steps = 16; shift = 10.0; CS = 3
rel = lambda p, q: float((p - q).norm() / q.norm())
g = torch.Generator(device=dev).manual_seed(7); z = torch.randn((16, 3) + tuple(lat[1:]), generator=g, device=dev)
E = torch.randn(pref.shape, generator=g, device=dev)
for tv in (999.0, 960.0, 900.0, 800.0, 600.0, 400.0):
    s_ = tv / 1000; ex = "high" if tv >= b1.boundary_t else "low"; m = M[ex]; kv = BaseKV(lambda: m, win=10 ** 6)
    pn = (1 - s_) * pref + s_ * E; zf = torch.cat([pn, z], 1)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        f = lambda ctx: m([zf.to(torch.bfloat16)], t=torch.full((1,), tv, device=dev), context=ctx, seq_len=c["seq_len"], y=[c["y"]], dit_cond_dict=c["dit_cond"])[0].float()[:, 3:]
        Fc, Fu = f([c["ctx"][0]]), f(c["ctxn"]); Fg = Fu + 5 * (Fc - Fu)
    res = []
    for tp in (0.0, 300.0, 600.0, 800.0, 900.0, 950.0, 980.0, 999.0):
        sp = tp / 1000; pp = (1 - sp) * pref + sp * E; kv.reset()
        kv.forward("c", pp, c["y"][:, :3], 0, tp, [c["ctx"][0]], cam[:, :, :3], write=True); kv.forward("u", pp, c["y"][:, :3], 0, tp, c["ctxn"], cam[:, :, :3], write=True)
        Kc = kv.forward("c", z, c["y"][:, 3:6], 3, tv, [c["ctx"][0]], cam[:, :, 3:6]); Ku = kv.forward("u", z, c["y"][:, 3:6], 3, tv, c["ctxn"], cam[:, :, 3:6])
        res.append("%d:%.2f" % (tp, rel(Ku + 5 * (Kc - Ku), Fg)))
    print("t", tv, ex, " ".join(res), flush=True)
print("ctxn", type(c["ctxn"]), len(c["ctxn"]), tuple(c["ctxn"][0].shape), "ctx", type(c["ctx"]), len(c["ctx"]), tuple(c["ctx"][0].shape))
sys.exit()
def run(mode):
    sch = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False); sch.set_timesteps(steps, device=dev, shift=shift)
    g = torch.Generator(device=dev).manual_seed(7); built = None; kv = BaseKV(lambda: CUR[0], win=10 ** 6)
    if mode == "full": z = torch.randn((16, 6) + tuple(lat[1:]), generator=g, device=dev)
    else: z = torch.randn((16, 3) + tuple(lat[1:]), generator=g, device=dev)
    for i, t in enumerate(sch.timesteps):
        tv = float(t); s = float(sch.sigmas[i]); ex = "high" if tv >= b1.boundary_t else "low"; CUR[0] = M[ex]
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if mode == "full":
                z[:, :3] = (1 - s) * pref + s * torch.randn(pref.shape, generator=g, device=dev)
                f = lambda ctx: M[ex]([z.to(torch.bfloat16)], t=torch.full((1,), tv, device=dev), context=ctx, seq_len=c["seq_len"], y=[c["y"]], dit_cond_dict=c["dit_cond"])[0].float()
                fc = f([c["ctx"][0]]); flow = f(c["ctxn"]); flow = flow + guide * (fc - flow)
            else:
                if built != ex:
                    kv.reset(); kv.forward("c", pref, c["y"][:, :3], 0, 0.0, [c["ctx"][0]], cam[:, :, :3], write=True); kv.forward("u", pref, c["y"][:, :3], 0, 0.0, c["ctxn"], cam[:, :, :3], write=True); built = ex
                fc = kv.forward("c", z, c["y"][:, 3:6], 3, tv, [c["ctx"][0]], cam[:, :, 3:6]); fu = kv.forward("u", z, c["y"][:, 3:6], 3, tv, c["ctxn"], cam[:, :, 3:6]); flow = fu + guide * (fc - fu)
        print(mode, i, round(tv), ex, "z %.3f flow %.3f" % (float(z.std()), float(flow.std())), flush=True)
        z = sch.step(flow.unsqueeze(0), t, z.unsqueeze(0), return_dict=False, generator=g)[0].squeeze(0)
    return z if mode == "kv" else z[:, 3:]
CUR = [None]; out = {}
for mode in ("kv", "full"):
    zz = run(mode)
    with torch.no_grad(): px = vae.decode([torch.cat([pref, zz], 1)])[0].clamp(-1, 1)
    out[mode] = ((px.permute(1, 2, 3, 0).float().cpu().numpy() + 1) * 127.5).astype(np.uint8)
row = lambda V: np.concatenate([cv2.resize(V[k], (256, 144)) for k in (0, 8, 12, 16, 20)], 1)
cv2.imwrite("/tmp/claude-1741758895/-home-hk28965/2af3c383-3218-4bda-a425-8a247f5f536a/scratchpad/kvtest.jpg", cv2.cvtColor(np.concatenate([row(out["kv"]), row(out["full"])], 0), cv2.COLOR_RGB2BGR))
