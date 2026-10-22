import os
import numpy as np, os, sys
S = os.environ["SCRATCH"]+""; sc, f0, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
src = "%s/traj55_action_%s" % (S, sc); d = "%s/traj55_slice%s_f%d_n%d" % (S, sc, f0, n); os.makedirs(d, exist_ok=True)
for f in ["poses.npy", "intrinsics.npy", "action.npy", "wasd_action.npy"]:
    a = np.load(os.path.join(src, f)); assert len(a) >= f0 + n, (f, len(a), f0 + n); np.save(os.path.join(d, f), a[f0:f0 + n].astype(np.float32))
print(d)
