#!/usr/bin/env python3
"""저해상도 롤아웃 → super-resolution (Lanczos / EDSR x2·x4 via super-image) → 480x832 로 맞춘 뒤 DINO 거리(Base bf16 480 참조, Fast14B 480 참조) + 시간 기록.
usage (sr-venv): python scripts/sr_eval.py  -> specwm_out/lowres_sr/sr_results.json / .md"""
import os, sys, json, time, glob, subprocess
import numpy as np, torch, imageio
from PIL import Image
S = "/home/hk28965/specwm"; R = S + "/lingbot-world-v2/specwm_out/lowres_sr"; TW, TH = 832, 464
def read(p):
    return [np.asarray(f) for f in imageio.get_reader(p)]
def write(p, frames): imageio.mimwrite(p, frames, fps=16, quality=8)
def lanczos(frames): return [np.asarray(Image.fromarray(f).resize((TW, TH), Image.LANCZOS)) for f in frames]
_models = {}
def edsr(frames, scale):
    from super_image import EdsrModel, ImageLoader
    if scale not in _models: _models[scale] = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=scale).cuda().eval()
    m = _models[scale]; out = []
    with torch.no_grad():
        for f in frames:
            x = ImageLoader.load_image(Image.fromarray(f)).cuda(); y = m(x)[0].clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
            out.append(np.asarray(Image.fromarray(y).resize((TW, TH), Image.LANCZOS)))
    return out
res = json.load(open(R + "/sr_results.json")) if os.path.exists(R + "/sr_results.json") else {}
srcs = {}
for d in sorted(glob.glob(R + "/*_*/")):
    mp4 = glob.glob(d + "*.mp4")
    if not mp4: continue
    srcs[os.path.basename(d.rstrip("/"))] = mp4[0]
for name, mp4 in srcs.items():
    fr = read(mp4); h, w = fr[0].shape[:2]; print(name, w, h, len(fr), flush=True)
    variants = {"lanczos": lambda: lanczos(fr)}
    if h < TH: variants["edsr_x2"] = lambda: edsr(fr, 2)
    if h * 2 < TH: variants["edsr_x4"] = lambda: edsr(fr, 4)
    for v, fn in variants.items():
        key = name + "__" + v; out = R + "/sr/" + key + ".mp4"; os.makedirs(R + "/sr", exist_ok=True)
        if os.path.exists(out) and key in res: continue
        torch.cuda.synchronize() if torch.cuda.is_available() else None; t = time.time(); up = fn(); torch.cuda.synchronize() if torch.cuda.is_available() else None; sr_s = time.time() - t
        write(out, up); np.save(R + "/sr/" + key + ".npy", np.stack(up[::4])); res[key] = dict(src=name, method=v, src_hw=[h, w], sr_s=round(sr_s, 2), frames=len(fr))
        json.dump(res, open(R + "/sr_results.json", "w"), indent=1); print(key, res[key], flush=True)
print("[SR_DONE]")
