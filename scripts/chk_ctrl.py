import os
import numpy as np, json
R = os.environ["SCRATCH"]+"/lingbot-world-v2/specwm_out"
a = np.load(R + "/e2e55/control/frames_all.npy"); b = np.load(R + "/resync55/W36_sink18/fast_alone_s1234/frames_all.npy"); n = min(len(a), len(b))
d = np.abs(a[:n].astype(int) - b[:n].astype(int)); print("E2E control vs continuous fast_alone: frames", a.shape, b.shape, "| max |diff| =", int(d.max()), "| mean", round(float(d.mean()), 4), "| first differing 4fps frame:", int(np.argmax(d.reshape(n, -1).max(1) > 0)) if d.max() > 0 else None)
