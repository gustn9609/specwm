"""Decode latent files (.pt, [16, F, h, w]) with the Wan2.1 VAE and write (1) an mp4 per file, (2) a labelled side-by-side mp4,
(3) a PNG grid of selected frames (rows = files, cols = frames). usage: python -m specwm.decode_cmp OUTPREFIX label=path.pt ..."""
import sys, os, numpy as np, torch, imageio
from PIL import Image, ImageDraw
sys.path.insert(0, "/home/hk28965/specwm/lingbot-world-v2")
from wan.modules.vae2_1 import Wan2_1_VAE
out = sys.argv[1]; items = [x.split("=", 1) for x in sys.argv[2:]]
vae = Wan2_1_VAE(vae_pth="/home/hk28965/specwm/lingbot-world-v2/lingbot-world-v2-14b-causal-fast/Wan2.1_VAE.pth", device="cuda:0")
vids = []
for lab, p in items:
    lat = torch.load(p).float().cuda()
    with torch.no_grad(): im = vae.decode([lat])[0].float().clamp(-1, 1)
    v = ((im + 1) * 127.5).permute(1, 2, 3, 0).byte().cpu().numpy(); vids.append(v)
    imageio.mimwrite(out + "_" + lab + ".mp4", list(v), fps=16, quality=8)
def lab_frame(f, s):
    im = Image.fromarray(f); d = ImageDraw.Draw(im); d.rectangle([0, 0, 8 + 9 * len(s), 22], fill=(0, 0, 0)); d.text((4, 5), s, fill=(255, 255, 0)); return np.array(im)
T = min(len(v) for v in vids)
imageio.mimwrite(out + "_side.mp4", [np.concatenate([lab_frame(v[t], l) for (l, _), v in zip(items, vids)], 1) for t in range(T)], fps=16, quality=8)
cols = [0, 40, 80, 120, T - 1]; sm = lambda f: np.array(Image.fromarray(f).resize((f.shape[1] // 2, f.shape[0] // 2)))
rows = [np.concatenate([sm(lab_frame(v[t], "%s f%d" % (l, t))) for t in cols], 1) for (l, _), v in zip(items, vids)]
Image.fromarray(np.concatenate(rows, 0)).save(out + "_grid.png"); print("ok", out)
