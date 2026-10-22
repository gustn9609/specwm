import os
os.environ["OMP_NUM_THREADS"] = "1"
import cv2; cv2.setNumThreads(1)
import numpy as np, imageio
S = os.environ["SCRATCH"]+""; R = S + "/lingbot-world-v2/specwm_out"
def read(p):
    c = cv2.VideoCapture(p); fr = []
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return np.stack(fr)
F = read(R + "/hybrid/hands/FAST_asis/video.mp4")
Rf = read(R + "/hybrid/hands/REFINE_0.40_0.20_0.08/video.mp4")
B = read(R + "/hands/base_p1/video.mp4")
n = min(len(F), len(Rf), len(B))
def label(img, txt):
    img = img.copy(); cv2.rectangle(img, (0, 0), (330, 34), (0, 0, 0), -1)
    cv2.putText(img, txt, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2); return img
frames = [np.concatenate([label(F[i], "FAST (18s)"), label(Rf[i], "FAST + Base refine (47s)"),
                          label(B[i], "BASE (816s)")], axis=1) for i in range(n)]
imageio.mimwrite(S + "/hands_3up.mp4", frames, fps=16, quality=8, codec="libx264", pixelformat="yuv420p")
print("hands_3up.mp4 저장:", frames[0].shape, n, "프레임")
