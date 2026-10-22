#!/usr/bin/env python3
"""Transcode comparison clips for the web page: 424 px wide, 8 fps, h264 crf ~30, faststart. Emits web/vid/<group>__<arm>.mp4 + manifest."""
import os, json, subprocess, numpy as np, imageio
S="/home/hk28965/specwm"; O=S+"/lingbot-world-v2/specwm_out"; ST=O+"/stevo"; W=S+"/specwm_out_web/vid"; os.makedirs(W, exist_ok=True); FF="/home/hk28965/.local/bin/ffmpeg"
def npy2mp4(p, dst):
    V=np.load(p); V=V if len(V)<100 else V   # frames_all.npy is already 4x subsampled (4 fps)
    imageio.mimwrite(dst, list(V), fps=4, quality=8)
items=[]
TASKS=["falling_glass","pouring_water_into_cup","rocket_launch","inflating_balloon"]
ARMS=[("base","Base 70-step (480p)",lambda t: ST+"/lingbot_base/%s/video.mp4"%t),("fast480","Fast 480p 4-step",lambda t: ST+"/lingbot_fast/%s.mp4"%t),("fast360_2step","Fast 360p 2-step",lambda t: ST+"/lingbot_fast_360_2step/%s.mp4"%t),
      ("fast360_2step_esrgan","Fast 360p 2-step + ESRGAN",lambda t: ST+"/lingbot_fast_360_2step_esrgan/%s.mp4"%t),("v2_1p3b","v2 1.3B",lambda t: ST+"/lingbot_v2_1p3b/%s.mp4"%t),("v2_pretrain","v2 causal-pretrain 14B",lambda t: ST+"/lingbot_v2_pretrain_360_173/%s.mp4"%t),
      ("h3world","H3-World",lambda t: ST+"/h3world_lingbotcam/%s.mp4"%t),("minimax_h3","MiniMax-H3 (curtain)",lambda t: ST+"/h3_video/%s_curtain.mp4"%t),("longlive","LongLive-2.0-5B (curtain)",lambda t: ST+"/longlive2_5b_video/%s_curtain.mp4"%t),
      ("reground_ours","Fast + ours (KV-rewrite)",lambda t: ST+"/lingbot_reground_kvrw_laundered/%s.mp4"%t),("reground_ideal","Fast + ideal regrounding",lambda t: ST+"/lingbot_reground_resync_k6/%s.mp4"%t)]
FL=[("S05","dragon"),("S01","stonehenge"),("S00","great wall"),("S02","urban")]
FARMS=[("base","Base 70-step",lambda sc: O+"/floor/base/%s_s1234/video.mp4"%sc),("fast480","Fast 480p 4-step",lambda sc: O+"/floor/fast/%s/W18_sink6/fast_alone_s1234/video.mp4"%sc),("fast360_2step","Fast 360p 2-step draft",lambda sc: O+"/draft360_reground/%s/W18_sink6/fast_alone_s1234/video.mp4"%sc),
       ("reground_ours","Fast + ours (KV-rewrite)",lambda sc: O+"/e1repro/%s/W18_sink6/kvrw_laundered_s1234/video.mp4"%sc),("reground_ideal","Fast + ideal regrounding",lambda sc: O+"/e1repro/%s/W18_sink6/resync_k6_s1234/video.mp4"%sc),
       ("anchor_restart","Fast prefix + restart from corrected anchor (C2)",lambda sc: O+"/c2_anchor/%s/composite_anchor_target480_s1234.mp4"%sc),("draft360_ours","Fast 360p 2-step + ours",lambda sc: O+"/draft360_reground/%s/W18_sink6/kvrw_laundered_s1234/video.mp4"%sc)]
jobs=[("stevo",t,k,l,fn(t)) for t in TASKS for k,l,fn in ARMS]+[("floor",sc,k,l,fn(sc)) for sc,_ in FL for k,l,fn in FARMS]
for grp,tid,key,lab,src in jobs:
    if not os.path.exists(src): print("[skip]",tid,key); continue
    dst=W+"/%s__%s__%s.mp4"%(grp,tid,key)
    if not os.path.exists(dst):
        subprocess.run([FF,"-y","-loglevel","error","-i",src,"-vf","fps=8,scale=360:-2","-c:v","libx264","-preset","slow","-crf","33","-pix_fmt","yuv420p","-movflags","+faststart","-an",dst],check=True)
    items.append(dict(group=grp,task=tid,arm=key,label=lab,file=os.path.basename(dst),kb=round(os.path.getsize(dst)/1024)))
json.dump(items,open(W+"/manifest.json","w"),indent=1); print(len(items),"clips, total %.1f MB"%(sum(i["kb"] for i in items)/1024))
