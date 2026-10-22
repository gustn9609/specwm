import os
"""BASE 를 새 기하 지표로 재서, Fast 와의 격차가 실제로 존재하는지 본다."""
import numpy as np, os, sys, glob
sys.path.insert(0, os.environ["SCRATCH"]+"")
from geo_metric import PatchMatcher, geo_consistency
S = os.environ["SCRATCH"]+""; R = S + "/lingbot-world-v2/specwm_out"
K4 = np.load(S + "/lingbot-world-v2/examples/00/intrinsics.npy")[0]
IDX = {"00": np.load(S+"/rt2_idx.npy"), "01": np.load(S+"/revisit_rt_idx_01.npy"),
       "05": np.load(S+"/revisit_rt_idx_05.npy")}
POS = {k: np.load(S+"/poses_%s.npy"%k) for k in ["00","01","05"]}
mt = PatchMatcher(hub=S+"/.torch_hub")
tg = []
for sc in ["00","01","05"]:
    for cand in ["%s/revisit/base_S%s/frames_all.npy" % (R, sc), "%s/revisit/base_rt2/frames_all.npy" % R if sc=="00" else None]:
        if cand and os.path.exists(cand): tg.append(("BASE_S%s" % sc, sc, cand)); break
out = {}
for tag, scene, p in tg:
    V = np.load(p); idx = IDX[scene]
    fi = np.arange(len(V))*4; fi = fi[fi < len(idx)]; V = V[:len(fi)]
    Ps = POS[scene][idx[fi]]
    H, W = V.shape[1], V.shape[2]; fx, fy, cx, cy = K4; sx, sy = W/832.0, H/480.0
    K = np.array([[fx*sx,0,cx*sx],[0,fy*sy,cy*sy],[0,0,1]])
    r = geo_consistency(V, Ps, K, mt, thr_px=8.0, max_pairs=400)
    out[tag] = r
    print("  %-14s %d프레임 %d쌍  평균inlier %.3f" % (tag, len(V), len(r), r[:,2].mean()), flush=True)
np.savez(S+"/geo_base.npz", **out)
print("저장 %d개" % len(out), flush=True)
