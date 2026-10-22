#!/usr/bin/env python3
"""데모 갤러리: specwm_out/demo/<name>/{base16,fast_alone,resync_k6}/video.mp4 -> GALLERY_<name>.mp4 (라벨 붙인 가로 병렬)
+ contact sheet PNG (0/25/50/75/100% 프레임) + VR/DINO 요약 표 (있으면). 사용: python demo_gallery.py [name ...]"""
import os, sys, json, glob
import numpy as np, imageio
from PIL import Image, ImageDraw
S = os.environ.get("SCRATCH", "/home/hk28965/specwm"); R = S + "/lingbot-world-v2/specwm_out"; D = R + "/demo"
ARMS = [("Base 16-step (cloud ref)", "base16"), ("Fast alone (edge)", "fast_alone"), ("Resync: Base ctx 0-5 + Fast", "resync_k6")]
vr = json.load(open(R + "/vr/vr_scores.json"))["videos"] if os.path.exists(R + "/vr/vr_scores.json") else {}
names = sys.argv[1:] or sorted(os.path.basename(p) for p in glob.glob(D + "/*") if os.path.isdir(p) and not os.path.basename(p).startswith("GALLERY"))
rows = ["| name | arm | frames | VR mean | flicker |", "|---|---|---|---|---|"]
for name in names:
    vids = [(lab, D + "/%s/%s/video.mp4" % (name, arm)) for lab, arm in ARMS if os.path.exists(D + "/%s/%s/video.mp4" % (name, arm))]
    if not vids: continue
    V = [np.stack(imageio.mimread(p, memtest=False))[:, :, :, :3] for _, p in vids]; n = min(len(v) for v in V); s = 2; out = []
    for i in range(n):
        tiles = []
        for (lab, _), v in zip(vids, V):
            im = Image.fromarray(v[i][::s, ::s]); d = ImageDraw.Draw(im); d.rectangle([0, 0, im.width, 22], fill=(0, 0, 0)); d.text((6, 4), lab, fill=(255, 255, 255)); tiles.append(np.asarray(im))
        out.append(np.concatenate(tiles, 1))
    imageio.mimwrite(D + "/GALLERY_%s.mp4" % name, out, fps=16, quality=8, output_params=["-threads", "2"])
    idx = [int(round(t)) for t in np.linspace(0, n - 1, 5)]
    sheet = np.concatenate([out[i] for i in idx], 0); imageio.imwrite(D + "/GALLERY_%s.png" % name, sheet)
    for (lab, p) in vids:
        arm = p.split("/")[-2]; tag = "demo/%s/%s" % (name, arm); e = vr.get(tag, {}); aj = os.path.join(os.path.dirname(p), "arm.json"); rj = os.path.join(os.path.dirname(p), "revisit.json")
        meta = json.load(open(aj)) if os.path.exists(aj) else (json.load(open(rj)) if os.path.exists(rj) else {})
        rows.append("| %s | %s | %d | %s | %s |" % (name, arm, n, ("%.4f" % e["mean_score"]) if e.get("status") == "done" else "-", meta.get("flicker", "-")))
    print("[gallery] %s: %s (%d frames)" % (name, [l for l, _ in vids], n), flush=True)
open(D + "/GALLERY.md", "w").write("# Demo gallery (diverse prompts)\n\n" + "\n".join(rows) + "\n\n각 GALLERY_<name>.mp4: " + " | ".join(l for l, _ in ARMS) + "\n")
print("\n".join(rows)); print("[DEMO_GALLERY_DONE]")
