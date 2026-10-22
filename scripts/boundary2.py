import os; os.environ["OMP_NUM_THREADS"] = "1"
import cv2; cv2.setNumThreads(1); import numpy as np
S = os.environ["SCRATCH"]+""; R = S + "/lingbot-world-v2/specwm_out"
def read(p):
    c = cv2.VideoCapture(p); fr = []
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return np.stack(fr)
V = {"Base16 own prefix, mask": "revisit/e3sep_self_mask", "Base16 other seed, mask": "revisit/e3sep_otherseed_mask", "C  Fast prefix, mask": "revisit/base_prefix_S05_L9", "Fast prefix, RePaint": "revisit/e3a_repaint_fast",
     "Base own prefix, RePaint": "revisit/e3a_repaint_self", "D  laundered, RePaint": "revisit/e3c_repaint_laundered", "D2 refined, MASK": "revisit/e3d_mask_refined", "causal12, RePaint": "revisit/e3b_causal12_repaint_fast", "A  Fast alone": "resync_v2/W18_sink6/fast_alone_s1234", "B  Base16 indep": "revisit/base_official_S05_s16"}
print("%-28s %8s %8s %10s %8s" % ("video (16 fps)", "jump32>33", "freeze", "sharp_gen", "bright"), "   jump = |f33-f32| / clip median ; freeze = mean diff f33..f38 / median (1 = normal motion)")
for k, p in V.items():
    v = read(R + "/" + p + "/video.mp4"); d = np.abs(v[1:].astype(np.float32) - v[:-1].astype(np.float32)).mean(axis=(1, 2, 3)); med = np.median(d)
    sh = np.mean([cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_32F).var() for f in v[36::8]])
    print("%-28s %8.2f %8.2f %10.0f %8.1f" % (k, d[32] / med, d[33:39].mean() / med, sh, v[36:].mean()))
