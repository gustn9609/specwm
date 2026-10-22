import os
import numpy as np, os, shutil
S = os.environ["SCRATCH"]+""
idx = np.load(S + "/traj60_idx.npy")
P = np.load(S + "/poses_00.npy")[idx].astype(np.float32)
src = S + "/rt2_action_00"
d = S + "/traj60_action"; os.makedirs(d, exist_ok=True)
np.save(d + "/poses.npy", P)
for f in ["intrinsics.npy", "action.npy", "wasd_action.npy"]:
    a = np.load(os.path.join(src, f))
    rep = np.repeat(a[:1], len(P), axis=0) if len(a) < len(P) else a[:len(P)]
    np.save(os.path.join(d, f), rep.astype(np.float32))
    print("  %-18s %s" % (f, rep.shape))
print("poses.npy", P.shape)
print("검증: 원본 자세와 일치?", np.allclose(P, np.load(S+"/poses_00.npy")[idx]))
