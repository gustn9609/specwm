"""[10/16 stage-1 diagnosis] decode resampled latents with the Wan2.1 VAE to measure what bicubic latent resampling does to the picture:
  down: the cloud's 704-grid clip downsampled to the device's 576 grid (upper bound of a refiner that works at the device grid)
  up:   the device's raw 576 draft upsampled to the 704 grid (what the refiner input looks like without any correction)
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.lat_resample_decode --src-hi DIR_hi --src-lo DIR_lo --out DIR --mode down|up --tasks a,b,..
(DIR_*/<task>/lat.pt; output <out>/<task>/spec.mp4)"""
import argparse, os, sys, torch, imageio, time, torch.nn.functional as F
sys.path.insert(0, "/home/hk28965/specwm/lingbot-world-v2")
from wan.modules.vae2_1 import Wan2_1_VAE
ap = argparse.ArgumentParser(); ap.add_argument("--src-hi", required=True); ap.add_argument("--src-lo", required=True); ap.add_argument("--out", required=True); ap.add_argument("--mode", required=True, choices=["down", "up"]); ap.add_argument("--tasks", required=True)
a = ap.parse_args(); vae = Wan2_1_VAE(vae_pth="/home/hk28965/specwm/lingbot-world-v2/lingbot-world-v2-14b-causal-fast/Wan2.1_VAE.pth", device="cuda:0")
def rs(x, hw): return F.interpolate(x.permute(1, 0, 2, 3), size=tuple(hw), mode="bicubic", align_corners=False).permute(1, 0, 2, 3).contiguous()
for t in a.tasks.split(","):
    od = a.out + "/" + t
    if os.path.exists(od + "/spec.mp4"): continue
    hi = torch.load("%s/%s/lat.pt" % (a.src_hi, t)).float(); lo = torch.load("%s/%s/lat.pt" % (a.src_lo, t)).float(); t0 = time.time()
    lat = rs(hi, lo.shape[2:]) if a.mode == "down" else rs(lo, hi.shape[2:])
    with torch.no_grad(): im = vae.decode([lat.cuda()])[0].float().clamp(-1, 1)
    v = ((im + 1) * 127.5).permute(1, 2, 3, 0).byte().cpu().numpy(); os.makedirs(od, exist_ok=True); imageio.mimwrite(od + "/spec.mp4", list(v), fps=16, quality=8)
    torch.save(lat.half(), od + "/lat.pt"); print("[RESAMPLE_DECODE] %s %s %s -> %s %.1fs" % (a.mode, t, tuple(hi.shape[2:]), tuple(lat.shape[2:]), time.time() - t0), flush=True)
print("[RESAMPLE_DECODE_DONE]", flush=True)
