"""Action item 6: cost of Base one-step checks, and whether several noised copies of a chunk can be checked in ONE batched forward.
Loads the Base low expert on one GPU, builds a prefix+chunk window at 480p (L prefix latents + 3), and times a forward for batch sizes B
(B noised copies at different sigmas or seeds stacked in the batch dimension). usage: CUDA_VISIBLE_DEVICES=g python -m specwm.onestep_profile [--prefix 9]"""
import argparse, os, sys, json, time, torch
S = "/home/hk28965/specwm"; ap = argparse.ArgumentParser(); ap.add_argument("--prefix", type=int, default=9); ap.add_argument("--batches", default="1,2,4,8"); ap.add_argument("--area", type=int, default=399360); ap.add_argument("--expert", default="low"); ap.add_argument("--cfg", type=int, default=0, help="1: two forwards (cond + uncond)"); ap.add_argument("--tag", default=""); a = ap.parse_args()
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world"); os.chdir(S + "/lingbot-world")
import importlib, importlib.util
spec = importlib.util.spec_from_file_location("specwm_v1", S + "/lingbot-world/specwm/__init__.py", submodule_search_locations=[S + "/lingbot-world/specwm"]); pkg = importlib.util.module_from_spec(spec); sys.modules["specwm_v1"] = pkg; spec.loader.exec_module(pkg)
V1Binding = importlib.import_module("specwm_v1.binding_v1").V1Binding
dev = torch.device("cuda:0"); b1 = V1Binding(S + "/lingbot-world/lingbot-world-base-cam", t5_cpu=True, device=str(dev)); LOW = (b1.pipe.low_noise_model if a.expert == "low" else b1.pipe.high_noise_model).to(dev); b1.pipe.high_noise_model = None; b1.pipe.low_noise_model = None; b1.device = dev; torch.cuda.empty_cache()
T = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}["sand_filling_up_jar"]
L = a.prefix; F = 4 * (L + 3 - 1) + 1; c = b1.prepare(T["image"], "", T["action"], frame_num=F, max_area=a.area); lat = torch.randn((16,) + tuple(c["lat"]), device=dev)
print("[profile] %s expert, cfg %d, area %d: window %d prefix + 3 chunk latents = %d frames, grid %s, %d tokens" % (a.expert, a.cfg, a.area, L, F, tuple(c["lat"]), c["seq_len"]), flush=True)
res = {}
for B in [int(x) for x in a.batches.split(",")]:
    xs = [lat.to(torch.bfloat16) for _ in range(B)]; ts = torch.full((B,), 800.0, device=dev) if B > 1 else torch.full((1,), 800.0, device=dev)
    ctx = [c["ctx"][0]] * B; ys = [c["y"]] * B; dc = c["dit_cond"]
    dcB = {k: (v * B if isinstance(v, list) else v) for k, v in dc.items()} if dc else dc
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(2): LOW(xs, t=ts, context=ctx, seq_len=c["seq_len"], y=ys, dit_cond_dict=dcB)   # warm-up
            torch.cuda.synchronize(); t0 = time.time(); n = 3
            for _ in range(n):
                LOW(xs, t=ts, context=ctx, seq_len=c["seq_len"], y=ys, dit_cond_dict=dcB)
                if a.cfg: LOW(xs, t=ts, context=[c["ctxn"][0]] * B, seq_len=c["seq_len"], y=ys, dit_cond_dict=dcB)
            torch.cuda.synchronize(); dt = (time.time() - t0) / n
        res[B] = dt; print("[profile] B=%d: %.2f s per forward (%.2f s per check), peak %.0f GB" % (B, dt, dt / B, torch.cuda.max_memory_allocated() / 2**30), flush=True)
    except Exception as e:
        print("[profile] B=%d failed: %s" % (B, str(e)[:120]), flush=True); torch.cuda.empty_cache()
json.dump(dict(res=res, expert=a.expert, cfg=a.cfg, area=a.area, prefix=L, tokens=c["seq_len"]), open(S + "/lingbot-world-v2/specwm_out/onestep_profile_%s_%s_cfg%d_a%d_p%d.json" % (a.tag, a.expert, a.cfg, a.area, L), "w"))
