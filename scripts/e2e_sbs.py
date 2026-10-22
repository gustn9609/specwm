import os; os.environ["OMP_NUM_THREADS"] = "1"
import cv2; cv2.setNumThreads(1); import numpy as np, imageio
S = os.environ["SCRATCH"]+""; R = S + "/lingbot-world-v2/specwm_out"; O = S + "/compare_videos"
def read(p):
    c = cv2.VideoCapture(p); fr = []
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return fr
def lab(img, t, sub=None):
    img = img.copy(); (w, h), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2); cv2.rectangle(img, (0,0), (w+16, h+16), (0,0,0), -1); cv2.putText(img, t, (8, h+8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    if sub: (w2,h2),_ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2); cv2.rectangle(img, (0,h+16), (w2+16, h+h2+32), (0,0,0), -1); cv2.putText(img, sub, (8, h+h2+24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,230,255), 2)
    return img
V = [("Fast alone (no cloud)", read(R + "/e2e55/control/video.mp4"), "5/10"), ("Base anchors from t=0 (upper bound)", read(R + "/resync55/W36_sink18/resync_k6_s1234/video.mp4"), "10/10"),
     ("SYSTEM fixed anchors (corr @4.5/19.5/34.5 s)", read(R + "/e2e55/fixed/video.mp4"), "7/10"), ("SYSTEM rolling anchors (anchor replaced each time)", read(R + "/e2e55/rolling/video.mp4"), "2/10")]
T = min(len(v) for _, v, _ in V); H, W0 = V[0][1][0].shape[:2]; th, tw = 232, 416
fr = []
for t in range(T):
    cells = [lab(cv2.resize(v[t], (tw, th), interpolation=cv2.INTER_AREA), n, s) for n, v, s in V]
    grid = np.concatenate([np.concatenate(cells[:2], 1), np.concatenate(cells[2:], 1)], 0)
    bar = np.zeros((34, grid.shape[1], 3), np.uint8); txt = "t = %5.2f s" % (t/16) + ("   <- CORRECTION" if any(abs(t/16 - c) < 0.35 for c in (4.5, 19.5, 34.5)) else "")
    cv2.putText(bar, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2); fr.append(np.concatenate([bar, grid], 0))
imageio.mimwrite(O + "/22_e2e_cloud_grounded_55s.mp4", fr, fps=16, quality=7, codec="libx264", pixelformat="yuv420p", output_params=["-threads","2"]); print("22 저장", fr[0].shape, T)
