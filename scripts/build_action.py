"""서버에서 실제 pose 기반 재방문 궤적 액션 디렉토리를 만든다."""
import numpy as np, os, sys, shutil
src, dst, traj = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(dst, exist_ok=True)
P = np.load(traj); n = len(P)
np.save(os.path.join(dst, "poses.npy"), P)
for f in ("intrinsics.npy", "action.npy", "wasd_action.npy"):
    p = os.path.join(src, f)
    if not os.path.exists(p): continue
    a = np.load(p)
    a = a[:n] if a.shape[0] >= n else np.concatenate([a, np.repeat(a[-1:], n-a.shape[0], 0)])
    np.save(os.path.join(dst, f), a)
print(f"[action] {dst}: {n} frames, files={sorted(os.listdir(dst))}")
