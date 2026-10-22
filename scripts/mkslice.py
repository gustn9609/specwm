import os
import numpy as np, os, sys
S = os.environ["SCRATCH"]+""; src = S + "/traj55_action_05"; f0 = int(sys.argv[1]); n = int(sys.argv[2]); d = "%s/traj55_slice_f%d_n%d" % (S, f0, n); os.makedirs(d, exist_ok=True)
for f in ["poses.npy", "intrinsics.npy", "action.npy", "wasd_action.npy"]:
    a = np.load(os.path.join(src, f)); assert len(a) >= f0 + n, (f, len(a), f0 + n); np.save(os.path.join(d, f), a[f0:f0 + n].astype(np.float32))
print(d)
