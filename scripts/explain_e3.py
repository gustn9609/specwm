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
    img = img.copy(); (w, h), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2); cv2.rectangle(img, (0, 0), (w + 16, h + 16), (0, 0, 0), -1); cv2.putText(img, t, (8, h + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    if sub: (w2, h2), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2); cv2.rectangle(img, (0, h + 16), (w2 + 16, h + h2 + 32), (0, 0, 0), -1); cv2.putText(img, sub, (8, h + h2 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 230, 255), 2)
    return img
def banner(t, W, txt, color):
    bar = np.zeros((40, W, 3), np.uint8); bar[:] = color; cv2.putText(bar, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2); return bar
FA = read(R + "/resync_v2/W18_sink6/fast_alone_s1234/video.mp4"); BI = read(R + "/revisit/base_official_S05_s16/video.mp4")
MK = read(R + "/revisit/base_prefix_S05_L9/video.mp4"); LR = read(R + "/revisit/e3c_repaint_laundered/video.mp4"); LA = read(R + "/hybrid/s05_launder/REFINE_0.40_0.20_0.08/video.mp4")
T = min(len(FA), len(BI), len(MK), len(LR)); fr = []
for t in range(T):
    ph = "phase 1: first 2.25 s = Fast's frames given to Base (prefix)" if t < 33 else "phase 2: Base generates the rest (2.25-10 s)"
    col = (60, 60, 60) if t < 33 else (20, 90, 60)
    a = lab(FA[t], "A. Fast alone (4-step, drifted)", "what the device shows"); b = lab(BI[t], "B. Base16 independent", "cloud, if it had no prefix: reference")
    c = lab(MK[t], "C. Base given Fast prefix, mask (orig E3)", "0.392 to B: drifts toward Fast's look"); d = lab(LR[t], "D. Base given LAUNDERED prefix + RePaint", "0.310 to B  (floor 0.284 = another seed)")
    if t < 33:
        for im in (c, d): cv2.rectangle(im, (2, 2), (im.shape[1] - 3, im.shape[0] - 3), (255, 80, 80), 4)
    top = np.concatenate([a, b], 1); bot = np.concatenate([c, d], 1); W = top.shape[1]
    fr.append(np.concatenate([banner(t, W, "t = %.2f s   %s" % (t / 16, ph), col), top, bot], 0))
imageio.mimwrite(O + "/19_explain_prefix_conditioning.mp4", fr, fps=16, quality=8, codec="libx264", pixelformat="yuv420p", output_params=["-threads", "2"]); print("19 저장", fr[0].shape, T)
# laundering 전후 (prefix 2.25 s 만 반복, 3배 느리게)
T2 = 33; fr2 = []
for rep in range(3):
    for t in range(T2):
        a = lab(FA[t], "Fast frame (drifted, plastic texture)"); b = lab(LA[t], "same frame after laundering (Base low-noise expert, 3 steps)"); c = lab(BI[t], "Base independent (texture reference)")
        fr2.append(np.concatenate([a, b, c], 1))
imageio.mimwrite(O + "/20_explain_laundering.mp4", fr2, fps=8, quality=8, codec="libx264", pixelformat="yuv420p", output_params=["-threads", "2"]); print("20 저장", fr2[0].shape, len(fr2))
