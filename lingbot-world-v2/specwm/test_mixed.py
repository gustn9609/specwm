"""unit test: BaseKV forward (no cache, all frames at once) must equal the Base model's own forward."""
import os, sys, json, torch
S = "/home/hk28965/specwm"; sys.path.insert(0, S + "/lingbot-world"); os.chdir(S + "/lingbot-world")
import importlib, importlib.util
spec = importlib.util.spec_from_file_location("specwm_v1", S + "/lingbot-world/specwm/__init__.py", submodule_search_locations=[S + "/lingbot-world/specwm"]); pkg = importlib.util.module_from_spec(spec); sys.modules["specwm_v1"] = pkg; spec.loader.exec_module(pkg)
import wan.modules.model as WM
def _sdpa(q, k, v, k_lens=None, **kw):
    return torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2).to(q.dtype)
WM.flash_attention = _sdpa
V1Binding = importlib.import_module("specwm_v1.binding_v1").V1Binding
dev = torch.device("cuda:0"); b1 = V1Binding(S + "/lingbot-world/lingbot-world-base-cam", t5_cpu=True, device=str(dev))
m = b1.pipe.high_noise_model.to(dev); b1.pipe.low_noise_model = None; b1.device = dev
T = [t for t in json.load(open(S + "/data/stevo/tasks_camera.json")) if t["id"] == "sand_filling_up_jar"][0]
c = b1.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=21, max_area=480 * 832)
sys.path.insert(0, S + "/lingbot-world-v2"); from specwm.base_mixed import mixed_forward, down, down_cam
g = torch.Generator(device=dev).manual_seed(0); x = torch.randn((16,) + tuple(c["lat"]), generator=g, device=dev); tv = 950.0
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    A = m([x.to(torch.bfloat16)], t=torch.full((1,), tv, device=dev), context=[c["ctx"][0]], seq_len=c["seq_len"], y=[c["y"]], dit_cond_dict=c["dit_cond"])[0].float()
cam = c["dit_cond"]["c2ws_plucker_emb"][0]; rel = lambda p, q: float((p - q).norm() / q.norm())
segs = [dict(x=x[:, :3], y=c["y"][:, :3], f0=0, k=1, cam=cam[:, :, :3]), dict(x=x[:, 3:], y=c["y"][:, 3:], f0=3, k=1, cam=cam[:, :, 3:])]
B = mixed_forward(m, segs, tv, [c["ctx"][0]], [0, 1]); print("k=1 split vs full: rel", rel(torch.cat(B, 1), A), flush=True)
segs = [dict(x=x[:, :3], y=c["y"][:, :3], f0=0, k=1, cam=cam[:, :, :3]), dict(x=down(x[:, 3:5], 2), y=down(c["y"][:, 3:5], 2), f0=3, k=2, cam=down_cam(cam[:, :, 3:5], 2)), dict(x=x[:, 5:], y=c["y"][:, 5:], f0=5, k=1, cam=cam[:, :, 5:])]
Cm = mixed_forward(m, segs, tv, [c["ctx"][0]], [2])[0]; print("k=2 middle: last latent rel vs full", rel(Cm, A[:, 5:]), tuple(Cm.shape), flush=True)
