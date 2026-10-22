"""VAE-encode ROOT/*/*/video.mp4 (832x480) -> lat.pt next to it (full-res latents for the shared-PCA path figures). usage: python -m specwm.reencode_lat ROOT"""
import sys, os, glob, numpy as np, torch, imageio
S = "/home/hk28965/specwm"; sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from wan.modules.vae2_1 import Wan2_1_VAE
vae = Wan2_1_VAE(vae_pth=S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast/Wan2.1_VAE.pth", device="cuda"); n = 0
for p in sorted(glob.glob(sys.argv[1] + "/*/*/video.mp4")):
    dst = os.path.dirname(p) + "/lat.pt"
    if os.path.exists(dst): continue
    fr = np.stack([np.asarray(f) for f in imageio.get_reader(p)]); v = torch.from_numpy(fr).permute(3, 0, 1, 2).float().div(127.5).sub(1)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16): lat = vae.encode([v.cuda()])[0].float()
    torch.save(lat.half().cpu(), dst); n += 1
print("[REENC_DONE] %d" % n)
