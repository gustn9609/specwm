"""Encode Base 70-step rollouts (training scenes S00/S01/S02/S05, NOT the test tasks) into latents at the Base working resolution.
Output: <out>/<scene>_s<seed>.pt = dict(lat=[16,41,h,w] bf16, scene=...). usage: CUDA_VISIBLE_DEVICES=g python -m specwm.mixlora_cache --out DIR"""
import argparse, os, sys, glob, numpy as np, torch, imageio, cv2
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world"); os.chdir(S + "/lingbot-world")
import importlib, importlib.util
spec = importlib.util.spec_from_file_location("specwm_v1", S + "/lingbot-world/specwm/__init__.py", submodule_search_locations=[S + "/lingbot-world/specwm"]); pkg = importlib.util.module_from_spec(spec); sys.modules["specwm_v1"] = pkg; spec.loader.exec_module(pkg)
V1Binding = importlib.import_module("specwm_v1.binding_v1").V1Binding
dev = torch.device("cuda:0"); b1 = V1Binding(S + "/lingbot-world/lingbot-world-base-cam", t5_cpu=True, device=str(dev)); vae = b1.pipe.vae
for sc in ("00", "01", "02", "05"):
    img = S + "/lingbot-world-v2/examples/%s/image.jpg" % sc; prompt = open(S + "/p_%s.txt" % sc).read().strip(); act = S + "/rt2_action_%s" % sc
    c = b1.prepare(img, prompt, act, frame_num=161, max_area=480 * 832); h, w = c["lat"][1:]
    for vp in sorted(glob.glob(S + "/lingbot-world-v2/specwm_out/floor/base/S%s_s*/video.mp4" % sc)):
        name = os.path.basename(os.path.dirname(vp)); op = os.path.join(a.out, name + ".pt")
        if os.path.exists(op): continue
        V = [np.asarray(f) for f in imageio.get_reader(vp)][:161]
        if len(V) < 161: continue
        fr = torch.from_numpy(np.stack([cv2.resize(f, (w * 8, h * 8), interpolation=cv2.INTER_AREA) for f in V])).float().div(127.5).sub(1).permute(3, 0, 1, 2).to(dev)
        with torch.no_grad(): lat = vae.encode([fr])[0]
        torch.save(dict(lat=lat.to(torch.bfloat16).cpu(), scene=sc), op); print("[cache]", name, tuple(lat.shape), flush=True)
print("[cache] done", flush=True)
