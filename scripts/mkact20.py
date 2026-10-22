import os
import numpy as np, os
S = os.environ["SCRATCH"]+""; idx = np.load(S + "/traj20_idx_05.npy"); P = np.load(S + "/poses_05.npy")[idx].astype(np.float32)
src = S + "/rt2_action_05"; d = S + "/traj20_action_05"; os.makedirs(d, exist_ok=True); np.save(d + "/poses.npy", P)
for f in ["intrinsics.npy", "action.npy", "wasd_action.npy"]:
    a = np.load(os.path.join(src, f)); rep = np.repeat(a[:1], len(P), axis=0) if len(a) < len(P) else a[:len(P)]; np.save(os.path.join(d, f), rep.astype(np.float32))
print("traj20_action_05:", P.shape, "일치:", np.allclose(P[:161], np.load(src + "/poses.npy")))
