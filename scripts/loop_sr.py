"""Per-frame super-resolution of low-res loop outputs: ROOT/<task>/<arm>_s<seed>/video.mp4 -> ROOT_sr/<task>/<arm>_s<seed>/video.mp4 at 832x480
(Real-ESRGAN realesr-general-x4v3 compact net, 4x then bicubic to the target, as in scripts/sr_apply.py). Writes sr_timing.json per run (ms/frame).
usage: CUDA_VISIBLE_DEVICES=g python scripts/loop_sr.py ROOT [--size 832,480]"""
import argparse, glob, json, os, time, numpy as np, imageio, torch, torch.nn as nn, torch.nn.functional as F
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--size", default="832,480"); a = ap.parse_args(); TW, TH = [int(v) for v in a.size.split(",")]
class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
        super().__init__(); self.upscale = upscale; self.body = nn.ModuleList([nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)])
        for _ in range(num_conv): self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1)); self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1)); self.upsampler = nn.PixelShuffle(upscale)
    def forward(self, x):
        out = x
        for m in self.body: out = m(out)
        return self.upsampler(out) + F.interpolate(x, scale_factor=self.upscale, mode="nearest")
model = SRVGGNetCompact().cuda().eval().half(); model.load_state_dict(torch.load("/home/hk28965/specwm/hf/realesrgan/realesr-general-x4v3.pth", map_location="cpu")["params"], strict=True)
root = a.root.rstrip("/"); out_root = root + "_sr"; n = 0
for p in sorted(glob.glob(root + "/*/*/video.mp4")):
    rel = os.path.relpath(os.path.dirname(p), root); d = os.path.join(out_root, rel); dst = os.path.join(d, "video.mp4")
    if os.path.exists(dst): continue
    os.makedirs(d, exist_ok=True); V = np.stack([np.asarray(f) for f in imageio.get_reader(p)]); out = []; t_sr = 0.0
    with torch.no_grad():
        for i in range(0, len(V), 8):
            x = torch.from_numpy(np.ascontiguousarray(V[i:i + 8])).cuda().permute(0, 3, 1, 2).half() / 255; torch.cuda.synchronize(); t = time.perf_counter()
            y = model(x).clamp(0, 1); y = F.interpolate(y, size=(TH, TW), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1); torch.cuda.synchronize(); t_sr += time.perf_counter() - t
            out.extend(list((y * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()))
    imageio.mimwrite(dst, out, fps=16, quality=8)
    lj = os.path.join(os.path.dirname(p), "loop.json"); meta = json.load(open(lj)) if os.path.exists(lj) else {}
    json.dump(dict(frames=len(V), src_hw=list(V.shape[1:3]), sr_s=round(t_sr, 2), ms_per_frame=round(1000 * t_sr / len(V), 2), gen_s=meta.get("gen_s"), plan=meta.get("plan")), open(os.path.join(d, "sr_timing.json"), "w")); n += 1
    if n % 10 == 0: print("[sr] %d done, last %s %.1f ms/frame" % (n, rel, 1000 * t_sr / len(V)), flush=True)
print("[SR_DONE] %s -> %s: %d videos" % (root, out_root, n), flush=True)
