import os
import numpy as np, glob, os, imageio
n = 0
for d in sorted(glob.glob(os.environ["SCRATCH"]+"/lingbot-world-v2/specwm_out/multi_scene/S*")):
    p, o = os.path.join(d, "frames_all.npy"), os.path.join(d, "video.mp4")
    if not os.path.exists(p) or os.path.exists(o): continue
    imageio.mimwrite(o, list(np.load(p)), fps=4, quality=8); n += 1
print("multi_scene 영상 %d개 생성" % n)
