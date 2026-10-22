#!/usr/bin/env python3
"""SR conditions for the 360p 2-step draft: lanczos (C) / esrgan (E: Real-ESRGAN realesr-general-x4v3, SRVGGNetCompact, 4x then Lanczos down to target).
Writes <out>/<task>.mp4 at target size + timing json (ms/frame). usage: sr_apply.py --src <dir> --out <dir> --method lanczos|esrgan [--size 848,464]"""
import argparse, glob, json, os, time, numpy as np, imageio, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True); ap.add_argument("--method", required=True); ap.add_argument("--size", default="848,464"); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); TW, TH = [int(v) for v in a.size.split(",")]
class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
        super().__init__(); self.upscale = upscale; self.body = nn.ModuleList([nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)])
        for _ in range(num_conv): self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1)); self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1)); self.upsampler = nn.PixelShuffle(upscale)
    def forward(self, x):
        out = x
        for m in self.body: out = m(out)
        return self.upsampler(out) + F.interpolate(x, scale_factor=self.upscale, mode="nearest")
model = None
if a.method == "esrgan":
    model = SRVGGNetCompact().cuda().eval().half(); model.load_state_dict(torch.load("/home/hk28965/specwm/hf/realesrgan/realesr-general-x4v3.pth", map_location="cpu")["params"], strict=True)
def sr_frames(V):
    out = []; t_sr = 0.0
    if a.method == "lanczos":
        for f in V:
            t = time.perf_counter(); out.append(np.asarray(Image.fromarray(f).resize((TW, TH), Image.LANCZOS))); t_sr += time.perf_counter() - t
    else:
        with torch.no_grad():
            for i in range(0, len(V), 8):
                x = torch.from_numpy(np.ascontiguousarray(V[i:i + 8])).cuda().permute(0, 3, 1, 2).half() / 255; torch.cuda.synchronize(); t = time.perf_counter()
                y = model(x).clamp(0, 1); y = F.interpolate(y, size=(TH, TW), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1); torch.cuda.synchronize(); t_sr += time.perf_counter() - t
                out.extend(list((y * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()))
    return out, t_sr
vids = sorted(glob.glob(a.src + "/*.mp4")); vids = vids[:a.limit] if a.limit else vids; rep = {}
for p in vids:
    n = os.path.basename(p); dst = os.path.join(a.out, n)
    if os.path.exists(dst): continue
    V = np.stack([np.asarray(f) for f in imageio.get_reader(p)]); out, t_sr = sr_frames(V)
    imageio.mimwrite(dst, out, fps=16, quality=8); rep[n] = dict(frames=len(V), src_hw=list(V.shape[1:3]), ms_per_frame=round(1000 * t_sr / len(V), 2)); print(n, rep[n], flush=True)
json.dump(rep, open(os.path.join(a.out, "sr_timing_%s.json" % a.method), "w"), indent=1)
ms = [r["ms_per_frame"] for r in rep.values()]; print("[SR_DONE] %s mean %.2f ms/frame over %d videos" % (a.method, np.mean(ms) if ms else -1, len(ms)))
