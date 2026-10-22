import os; os.environ["OMP_NUM_THREADS"] = "1"
import cv2; cv2.setNumThreads(1); import numpy as np, imageio
S = os.environ["SCRATCH"]+""; R = S + "/lingbot-world-v2/specwm_out/aria"; O = S + "/aria/deliver"; os.makedirs(O, exist_ok=True)
def read(p, n=None):
    c = cv2.VideoCapture(p); fr = []
    while n is None or len(fr) < n:
        ok, f = c.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return fr
def lab(img, t):
    img = img.copy(); (w, h), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2); cv2.rectangle(img, (0, 0), (w + 16, h + 16), (0, 0, 0), -1); cv2.putText(img, t, (8, h + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2); return img
wr = lambda p, fr: imageio.mimwrite(p, fr, fps=16, quality=8, codec="libx264", pixelformat="yuv420p", output_params=["-threads", "2"])
for W, sd in [("E_spatula_to_plate", 1234), ("B2_pour_liquid_pan", 1234), ("D_wash_tomato", 1234)]:
    G = read(S + "/aria/win_%s/gt_161f_16fps.mp4" % W, 81); B = read(R + "/%s_base70_81f_s%d/video.mp4" % (W, sd)); T = min(len(G), len(B))
    wr(O + "/GT_%s_5s.mp4" % W, G[:T]); wr(O + "/GEN_base70_%s_5s_s%d.mp4" % (W, sd), B[:T])
    wr(O + "/SIDEBYSIDE_%s_5s_s%d.mp4" % (W, sd), [np.concatenate([lab(G[t], "Ground truth (Project Aria, AEA)"), lab(B[t], "LingBot-World Base, 70 steps, seed %d" % sd)], 1) for t in range(T)]); print(W, sd, T, flush=True)
