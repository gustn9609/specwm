"""Low-res counterparts for the resolution-split refiner: for each res_data clip (14B 832x480 rollout, lat.pt [16,F,58,104]) decode with the VAE,
downscale the video to --lo (576x320), VAE-encode again -> lo.pt [16,F,40,72] fp16 (what the cloud would send), plus up.pt = lo latent bilinearly
resampled to the 832 grid (the refiner's input). usage: CUDA_VISIBLE_DEVICES=g python -m specwm.res_pairs --root specwm_out/res_data [--shard i/n]"""
import argparse, os, sys, glob, torch, torch.nn.functional as F
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--lo", default="576x320"); ap.add_argument("--shard", default="0/1"); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from wan.modules.vae2_1 import Wan2_1_VAE
vae = Wan2_1_VAE(vae_pth=S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast/Wan2.1_VAE.pth", device="cuda"); W, H = [int(v) for v in a.lo.split("x")]; si, sn = [int(v) for v in a.shard.split("/")]
clips = sorted(d for d in glob.glob(a.root + "/*_s*") if os.path.exists(d + "/lat.pt"))[si::sn]; n = 0
for d in clips:
    if os.path.exists(d + "/up.pt"): continue
    L = torch.load(d + "/lat.pt").float().cuda()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        v = vae.decode([L])[0].float().clamp(-1, 1)                                   # [3, T, Hv, Wv]
        v_lo = F.interpolate(v.permute(1, 0, 2, 3), size=(H, W), mode="bicubic", align_corners=False, antialias=True).clamp(-1, 1).permute(1, 0, 2, 3)
        lo = vae.encode([v_lo])[0].float()                                            # [16, F, h_lo, w_lo]
    up = F.interpolate(lo.permute(1, 0, 2, 3), size=tuple(L.shape[2:]), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
    torch.save(lo.half().cpu(), d + "/lo.pt"); torch.save(up.half().cpu(), d + "/up.pt"); n += 1
    if n % 10 == 0: print("[pairs] %d %s lo %s" % (n, os.path.basename(d), tuple(lo.shape)), flush=True)
print("[PAIRS_DONE] %d" % n, flush=True)
