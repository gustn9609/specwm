import os
"""기존 arm 전체에 기하 일관성 지표를 적용한다."""
import numpy as np, glob, os, sys, json
sys.path.insert(0, os.environ["SCRATCH"]+"")
from geo_metric import PatchMatcher, geo_consistency
S = os.environ["SCRATCH"]+""
R = S + "/lingbot-world-v2/specwm_out"
K4 = np.load(S + "/lingbot-world-v2/examples/00/intrinsics.npy")[0]
IDX = {"00": np.load(S+"/rt2_idx.npy"), "01": np.load(S+"/revisit_rt_idx_01.npy"),
       "05": np.load(S+"/revisit_rt_idx_05.npy")}
POS = {k: np.load(S+"/poses_%s.npy"%k) for k in ["00","01","05"]}
mt = PatchMatcher(hub=S+"/.torch_hub")
out = {}
targets = sorted(glob.glob(R+"/multi_scene/S*")) + sorted(glob.glob(R+"/win_sweep2/W*"))
for d in targets:
    tag = os.path.basename(d)
    p = os.path.join(d, "frames_all.npy")
    if not os.path.exists(p): continue
    scene = tag[1:3] if tag.startswith("S") else "00"
    V = np.load(p); idx = IDX[scene]
    fi = np.arange(len(V))*4; fi = fi[fi < len(idx)]; V = V[:len(fi)]
    # 각 저장프레임에 대응하는 원본 자세
    Ps = POS[scene][idx[fi]]
    # 해상도에 맞춘 K (원본 832x480 기준 -> 실제 프레임 크기로 스케일)
    H, W = V.shape[1], V.shape[2]
    fx, fy, cx, cy = K4
    sx, sy = W/832.0, H/480.0
    K = np.array([[fx*sx,0,cx*sx],[0,fy*sy,cy*sy],[0,0,1]])
    r = geo_consistency(V, Ps, K, mt, thr_px=8.0, max_pairs=400)
    out[tag] = r
    print("  %-28s %d쌍  평균inlier %.3f" % (tag, len(r), r[:,2].mean() if len(r) else -1), flush=True)
np.savez(S+"/geo_pairs.npz", **out)
print("저장 %d arm" % len(out), flush=True)
