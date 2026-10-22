import os
import numpy as np, os, sys, glob
sys.path.insert(0, os.environ["SCRATCH"]+"")
from geo_metric import PatchMatcher, geo_consistency
S = os.environ["SCRATCH"]+""; R = S + "/lingbot-world-v2/specwm_out/win60"
K4 = np.load(S + "/lingbot-world-v2/examples/00/intrinsics.npy")[0]
idx = np.load(S + "/traj60_idx.npy"); P = np.load(S + "/poses_00.npy")
mt = PatchMatcher(hub=S + "/.torch_hub"); out = {}
for d in sorted(glob.glob(R + "/W*")):
    tag = os.path.basename(d); p = os.path.join(d, "frames_all.npy")
    if not os.path.exists(p): continue
    V = np.load(p); fi = np.arange(len(V)) * 4; fi = fi[fi < len(idx)]; V = V[:len(fi)]
    Ps = P[idx[fi]]; H, W = V.shape[1], V.shape[2]; fx, fy, cx, cy = K4; sx, sy = W / 832.0, H / 480.0
    K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]])
    r = geo_consistency(V, Ps, K, mt, thr_px=8.0, max_pairs=3000)
    out[tag] = r; print("  %-8s %d프레임 %d쌍 inlier %.3f" % (tag, len(V), len(r), r[:, 2].mean()), flush=True)
np.savez(S + "/geo_win60.npz", **out); print("저장 %d arm" % len(out), flush=True)
