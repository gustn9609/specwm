import os
import numpy as np, os
S = os.environ["SCRATCH"]+""; idx = np.load(S + "/traj55_idx_05.npy"); P = np.load(S + "/poses_05.npy")[idx].astype(np.float32)
src = S + "/rt2_action_05"; d = S + "/traj55_action_05"; os.makedirs(d, exist_ok=True); np.save(d + "/poses.npy", P)
for f in ["intrinsics.npy", "action.npy", "wasd_action.npy"]:
    a = np.load(os.path.join(src, f)); rep = np.repeat(a[:1], len(P), axis=0) if len(a) < len(P) else a[:len(P)]; np.save(os.path.join(d, f), rep.astype(np.float32))
print("traj55_action_05:", P.shape, "첫161 일치:", np.allclose(P[:161], np.load(src + "/poses.npy")))
# rt2_action_02: examples/02 의 첫 161 프레임
e2 = S + "/lingbot-world-v2/examples/02"; d2 = S + "/rt2_action_02"; os.makedirs(d2, exist_ok=True)
for f in ["poses.npy", "intrinsics.npy", "action.npy", "wasd_action.npy"]:
    a = np.load(os.path.join(e2, f)); np.save(os.path.join(d2, f), (a[:161] if len(a) >= 161 else np.repeat(a[:1], 161, axis=0)).astype(np.float32)); print(" ", f, a.shape, "->", np.load(os.path.join(d2, f)).shape)
open(S + "/p_02.txt", "w").write(open(e2 + "/prompt.txt").read().strip()); print("p_02.txt 저장")
