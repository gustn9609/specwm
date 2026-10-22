"""Decoder ablation on finished cloud latents: TAEHV (taew2_1) vs full Wan2.1 VAE. Writes <dir>/taehv.mp4 next to each lat.pt and times both
decoders per chunk (13 chunks/clip; full VAE timed on the first --vae-n clips). usage: python scripts/taehv_ablation.py GLOB [--vae-n 3]"""
import sys, glob, time, json, os, torch, imageio, numpy as np
S = "/home/hk28965/specwm"; sys.path.insert(0, S + "/tools/taehv"); sys.path.insert(0, S + "/lingbot-world-v2")
from taehv import TAEHV
g = sys.argv[1]; vae_n = int(sys.argv[sys.argv.index("--vae-n") + 1]) if "--vae-n" in sys.argv else 3; dev = "cuda"
tae = TAEHV(S + "/tools/taehv/taew2_1.pth").to(dev).half().eval()
from wan.modules.vae2_1 import Wan2_1_VAE
vae = Wan2_1_VAE(vae_pth=S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast/Wan2.1_VAE.pth", device=dev)
tt, tv = [], []; fs = sorted(glob.glob(g))
with torch.no_grad():
    for i, f in enumerate(fs):
        lat = torch.load(f).to(dev); torch.cuda.synchronize(); t0 = time.time()
        v = tae.decode_video(lat.permute(1, 0, 2, 3)[None].half(), parallel=True, show_progress_bar=False)[0]; torch.cuda.synchronize(); tt.append((time.time() - t0) / 13)
        imageio.mimwrite(os.path.dirname(f) + "/taehv.mp4", list((v.float().clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()), fps=16, quality=8)
        if i < vae_n:
            torch.cuda.synchronize(); t0 = time.time()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16): vae.decode([lat.float()])
            torch.cuda.synchronize(); tv.append((time.time() - t0) / 13)
print(json.dumps(dict(clips=len(fs), shape=list(lat.shape), taehv_ms_per_chunk=round(1000 * float(np.median(tt[1:])), 1), vae_ms_per_chunk=round(1000 * float(np.median(tv[1:])), 1))))
