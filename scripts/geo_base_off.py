import os
import numpy as np, os, sys
sys.path.insert(0, os.environ["SCRATCH"]+"")
from geo_metric import PatchMatcher, geo_consistency
S = os.environ["SCRATCH"]+""; R = S + "/lingbot-world-v2/specwm_out"
K4 = np.load(S + "/lingbot-world-v2/examples/00/intrinsics.npy")[0]
mt = PatchMatcher(hub=S + "/.torch_hub"); out = {}
for tag, scene, p in [("BASE_OFF_S05", "05", R + "/revisit/base_official_S05/frames_all.npy"), ("BASE_OFF_S01", "01", R + "/revisit/base_official_S01/frames_all.npy")]:
    idx = np.load(S + "/revisit_rt_idx_%s.npy" % scene); P = np.load(S + "/poses_%s.npy" % scene)
    V = np.load(p); fi = np.arange(len(V)) * 4; fi = fi[fi < len(idx)]; V = V[:len(fi)]
    Ps = P[idx[fi]]; H, W = V.shape[1], V.shape[2]; fx, fy, cx, cy = K4; sx, sy = W / 832.0, H / 480.0
    K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]])
    r = geo_consistency(V, Ps, K, mt, thr_px=8.0, max_pairs=1200); out[tag] = r
    print("  %-14s %d프레임 %d쌍 inlier %.4f" % (tag, len(V), len(r), r[:, 2].mean()), flush=True)
np.savez(S + "/geo_base_off.npz", **out); print("저장", flush=True)
