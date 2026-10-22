import os
import imageio, numpy as np, glob, os
d=os.environ["SCRATCH"]+"/lingbot-world-v2/specwm_out/vid_int8"
out=os.environ["SCRATCH"]+"/frames"; os.makedirs(out, exist_ok=True)
picks=[10,40,70]
for f in sorted(glob.glob(d+"/*.mp4")):
    key=os.path.basename(f)[:-4]
    rd=imageio.get_reader(f)
    for i,fr in enumerate(rd):
        if i in picks:
            imageio.imwrite(f"{out}/{key}__f{i}.png", fr)
        if i>max(picks): break
    rd.close()
    print("ok", key)
