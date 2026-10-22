import os; os.environ["OMP_NUM_THREADS"] = "1"
import cv2; cv2.setNumThreads(1); import numpy as np, imageio, sys
S = os.environ["SCRATCH"]+""; R = S + "/lingbot-world-v2/specwm_out/epic"; OUT = S + "/compare_videos"
def read(p):
    c = cv2.VideoCapture(p); fr = []
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return fr
def lab(img, t):
    img = img.copy(); (w, h), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2); cv2.rectangle(img, (0, 0), (w + 16, h + 16), (0, 0, 0), -1); cv2.putText(img, t, (8, h + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2); return img
for W in sys.argv[1:]:
    G = read(S + "/epic/%s/gt_161f_16fps.mp4" % W); B = read(R + "/%s_base70/video.mp4" % W); T = min(len(G), len(B))
    fr = [np.concatenate([lab(G[t], "Ground truth (EPIC-KITCHENS P28_101)"), lab(B[t], "LingBot-World Base, 70 steps, same first frame + camera")], 1) for t in range(T)]
    imageio.mimwrite(OUT + "/18_epic_hands_%s_gt_vs_base.mp4" % W, fr, fps=16, quality=8, codec="libx264", pixelformat="yuv420p", output_params=["-threads", "2"]); print("저장", W, fr[0].shape, T, flush=True)
