"""Action item 6: cost grid of ONE Base check (batch 1) over expert x CFG x resolution x prefix window, both experts loaded once.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.onestep_grid [--areas 399360,184320 --prefixes 3,9,18 --reps 3]"""
import argparse, os, sys, json, time, torch
S = "/home/hk28965/specwm"; ap = argparse.ArgumentParser(); ap.add_argument("--areas", default="399360,184320"); ap.add_argument("--prefixes", default="3,9,18"); ap.add_argument("--reps", type=int, default=3); a = ap.parse_args()
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world"); os.chdir(S + "/lingbot-world")
import importlib, importlib.util
spec = importlib.util.spec_from_file_location("specwm_v1", S + "/lingbot-world/specwm/__init__.py", submodule_search_locations=[S + "/lingbot-world/specwm"]); pkg = importlib.util.module_from_spec(spec); sys.modules["specwm_v1"] = pkg; spec.loader.exec_module(pkg)
V1Binding = importlib.import_module("specwm_v1.binding_v1").V1Binding
dev = torch.device("cuda:0"); b1 = V1Binding(S + "/lingbot-world/lingbot-world-base-cam", t5_cpu=True, device=str(dev)); EXP = {"high": b1.pipe.high_noise_model.to(dev), "low": b1.pipe.low_noise_model.to(dev)}; b1.device = dev; torch.cuda.empty_cache()
T = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}["sand_filling_up_jar"]; rows = []
for area in [int(x) for x in a.areas.split(",")]:
    for L in [int(x) for x in a.prefixes.split(",")]:
        F = 4 * (L + 3 - 1) + 1; c = b1.prepare(T["image"], "", T["action"], frame_num=F, max_area=area); lat = torch.randn((16,) + tuple(c["lat"]), device=dev).to(torch.bfloat16)
        for ex in ("high", "low"):
            m = EXP[ex]; tv = 980.0 if ex == "high" else 800.0; tt = torch.full((1,), tv, device=dev)
            for cfg in (0, 1):
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    def call():
                        m([lat], t=tt, context=[c["ctx"][0]], seq_len=c["seq_len"], y=[c["y"]], dit_cond_dict=c["dit_cond"])
                        if cfg: m([lat], t=tt, context=c["ctxn"], seq_len=c["seq_len"], y=[c["y"]], dit_cond_dict=c["dit_cond"])
                    call(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
                    for _ in range(a.reps): call()
                    torch.cuda.synchronize(); dt = (time.time() - t0) / a.reps
                r = dict(area=area, grid=list(c["lat"]), prefix=L, latents=L + 3, tokens=c["seq_len"], expert=ex, cfg=cfg, sec=round(dt, 3), peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1)); rows.append(r)
                print("[grid] area %6d grid %-14s window %2d lat (%5d tok) %-4s cfg %d: %.2f s  peak %.0f GB" % (area, tuple(c["lat"]), L + 3, c["seq_len"], ex, cfg, dt, r["peak_gb"]), flush=True)
json.dump(rows, open(S + "/lingbot-world-v2/specwm_out/onestep_grid.json", "w"), indent=1); print("[grid] done", flush=True)
