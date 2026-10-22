import os
import numpy as np, imageio, sys
from PIL import Image, ImageDraw
R=os.environ["SCRATCH"]+"/lingbot-world-v2/specwm_out"; S=sys.argv[1]
def rd(p): return np.stack(imageio.mimread(p, memtest=False))[:, :, :, :3]
vids=[("Fast alone",R+"/basepatch/%s/fast_alone/video.mp4"%S),("Hybrid tau=0.20",R+"/basepatch/%s/hyb_t020/video.mp4"%S),("Base-refined (all Base patches)",R+"/basepatch/%s/base_refined/video.mp4"%S),("Base 16-step from scratch",R+"/revisit/base_official_S05_s16/video.mp4")]
V=[rd(p) for _,p in vids]; n=min(len(v) for v in V); s=2; out=[]
for i in range(n):
    tiles=[]
    for (lab,_),v in zip(vids,V):
        im=Image.fromarray(v[i][::s,::s]); d=ImageDraw.Draw(im); d.rectangle([0,0,im.width,22],fill=(0,0,0)); d.text((6,4),lab,fill=(255,255,255)); tiles.append(np.asarray(im))
    out.append(np.concatenate([np.concatenate(tiles[:2],1),np.concatenate(tiles[2:],1)],0))
imageio.mimwrite(os.environ["SCRATCH"]+"/SIDEBYSIDE_%s.mp4"%S, out, fps=16, quality=8, output_params=["-threads","2"]); print("saved",len(out),out[0].shape)
