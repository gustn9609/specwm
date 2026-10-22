import os; os.environ["OMP_NUM_THREADS"] = "1"
import cv2; cv2.setNumThreads(1); import numpy as np, imageio, sys
SC = sys.argv[1]; S = os.environ["SCRATCH"]+""; R = S + "/lingbot-world-v2/specwm_out"; O = S + "/compare_videos"
W36 = R + "/scene55/S%s/W36_sink18" % SC; E = R + "/e2e55_S%s" % SC
NAME = {"01": "Stonehenge", "02": "city walk", "05": "dragon over jungle"}[SC]
def read(p):
    c = cv2.VideoCapture(p); fr = []
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return fr
TH, TW = 240, 430
def tile(img, title, sub, badge=None, bad=False):
    im = cv2.resize(img, (TW, TH), interpolation=cv2.INTER_AREA)
    (w, h), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(im, (0, 0), (w + 12, h + 12), (0, 0, 0), -1); cv2.putText(im, title, (6, h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    if sub:
        (w2, h2), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(im, (0, h + 12), (w2 + 12, h + h2 + 24), (0, 0, 0), -1); cv2.putText(im, sub, (6, h + h2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 220, 250), 1)
    if badge:
        (w3, h3), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        col = (150, 150, 255) if bad else (120, 230, 160)
        cv2.rectangle(im, (TW - w3 - 14, TH - h3 - 14), (TW, TH), (0, 0, 0), -1); cv2.putText(im, badge, (TW - w3 - 7, TH - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    return im
P = [("Fast alone (no cloud)", "device only", W36 + "/fast_alone_s1234/video.mp4", None, False),
     ("Naive: polish own frames", "no new content from the cloud", W36 + "/kvrw_naive_s1234/video.mp4", None, False),
     ("One correction @4.5 s, pinned", "cloud regenerates 2.25 s, then nothing", W36 + "/kvrw_system_s1234/video.mp4", None, False),
     ("3 corrections, anchors FIXED", "4.5 / 19.5 / 34.5 s; first one stays pinned", E + "/fixed/video.mp4", None, False),
     ("3 corrections, anchors ROLLING", "same corrections, anchors overwritten", E + "/rolling/video.mp4", None, True),
     ("Upper bound", "cloud frames pinned from t=0", W36 + "/resync_k6_s1234/video.mp4", None, False)]
V = []
for n, sub, p, bg, bad in P:
    fr = read(p); print(n, len(fr), flush=True); V.append((n, sub, fr, bg, bad))
T = max(len(v[2]) for v in V)
wr = imageio.get_writer(O + "/25_S%s_anchor_policy_55s.mp4" % SC, fps=16, quality=7, codec="libx264", pixelformat="yuv420p", macro_block_size=None, output_params=["-threads", "2"])
for t in range(T):
    cells = [tile(fr[min(t, len(fr) - 1)], n, sub, bg, bad) for n, sub, fr, bg, bad in V]
    grid = np.concatenate([np.concatenate(cells[:3], 1), np.concatenate(cells[3:], 1)], 0)
    bar = np.zeros((36, grid.shape[1], 3), np.uint8)
    corr = any(abs(t / 16 - c) < 0.4 for c in (4.5, 19.5, 34.5))
    cv2.putText(bar, "%s   t = %5.2f s%s" % (NAME, t / 16, "    <<< CLOUD CORRECTION" if corr else ""), (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (90, 255, 160) if corr else (255, 255, 255), 2)
    wr.append_data(np.concatenate([bar, grid], 0))
wr.close(); print("[SBS_ANCHOR_DONE]", SC, T, flush=True)
