"""Train-time hints that match the TEST-time plan transform. For every cached Base rollout (mixlora/cache/*.pt, lat [16,41,h,w] at 480p):
decode -> resize to the plan resolution (area --area, same aspect, multiple of 16, INTER_AREA) -> (lo) VAE-encode at that resolution
(latent [16,41,hl,wl], to be bilinearly upsampled to (h,w) at train time) and (pix) cubic-upscale back to 480p -> VAE-encode ([16,41,h,w]).
Saves <out>/<name>.pt = dict(hint_lo, hint_pix, size_lo). usage: CUDA_VISIBLE_DEVICES=g python -m specwm.hint_cache [--area 184320]"""
import argparse, os, sys, glob, numpy as np, torch, cv2
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--cache", default=S + "/lingbot-world-v2/specwm_out/mixlora/cache"); ap.add_argument("--out", default=S + "/lingbot-world-v2/specwm_out/mixlora/hints")
ap.add_argument("--area", type=int, default=184320); ap.add_argument("--shard", default="0/1"); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from wan.modules.vae2_1 import Wan2_1_VAE
dev = torch.device("cuda:0"); vae = Wan2_1_VAE(vae_pth=S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast/Wan2.1_VAE.pth", device=dev)
k, n = [int(x) for x in a.shard.split("/")]
for i, f in enumerate(sorted(glob.glob(a.cache + "/*.pt"))):
    if i % n != k: continue
    op = os.path.join(a.out, os.path.basename(f))
    if os.path.exists(op): continue
    lat = torch.load(f)["lat"].to(dev).float()
    with torch.no_grad(): px = vae.decode([lat])[0].clamp(-1, 1)   # [3,161,H,W]
    C, F, H, W = px.shape; asp = H / W; hl = int(round(np.sqrt(a.area * asp) / 16)) * 16; wl = int(round(np.sqrt(a.area / asp) / 16)) * 16
    fr = ((px.permute(1, 2, 3, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    lo = np.stack([cv2.resize(x, (wl, hl), interpolation=cv2.INTER_AREA) for x in fr]); up = np.stack([cv2.resize(x, (W, H), interpolation=cv2.INTER_CUBIC) for x in lo])
    t = lambda v: torch.from_numpy(v).float().div(127.5).sub(1).permute(3, 0, 1, 2).to(dev)
    with torch.no_grad(): h_lo = vae.encode([t(lo)])[0].float(); h_pix = vae.encode([t(up)])[0].float()
    torch.save(dict(hint_lo=h_lo.to(torch.bfloat16).cpu(), hint_pix=h_pix.to(torch.bfloat16).cpu(), size_lo=(hl, wl)), op)
    print("[hint] %s lat %s lo %s pix %s" % (os.path.basename(f), tuple(lat.shape), tuple(h_lo.shape), tuple(h_pix.shape)), flush=True)
print("[hint] done", flush=True)
