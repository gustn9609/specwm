"""Decode draft_rollout.py latents to videos for blind rating: SRC/<task>__s<seed>.pt -> ROOT/<task>/<arm>_s<seed>/video.mp4 (16 fps, 153 frames).
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.rollout_decode --src DIR --arm NAME --root ROOT [--shard i/n]"""
import argparse, os, sys, glob, torch, imageio
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--arm", required=True); ap.add_argument("--root", required=True); ap.add_argument("--shard", default="0/1"); a = ap.parse_args()
si, sn = (int(x) for x in a.shard.split("/")); sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from wan.modules.vae2_1 import Wan2_1_VAE
dev = torch.device("cuda:0"); vae = Wan2_1_VAE(vae_pth=S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast/Wan2.1_VAE.pth", device=dev)
for f in sorted(glob.glob(a.src + "/*.pt"))[si::sn]:
    d = torch.load(f, map_location="cpu"); out = "%s/%s/%s_s%d" % (a.root, d["task"], a.arm, d["seed"])
    if os.path.exists(out + "/video.mp4"): continue
    os.makedirs(out, exist_ok=True)
    with torch.no_grad(): im = vae.decode([d["lat"].to(dev).float()])[0].float()
    fr = ((im.clamp(-1, 1) + 1) * 127.5).permute(1, 2, 3, 0).byte().cpu().numpy()[:153]; imageio.mimwrite(out + "/video.mp4", list(fr), fps=16, quality=8)
print("[DECODE_DONE]", a.src, a.shard, flush=True)
