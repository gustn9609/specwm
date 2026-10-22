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
TH, TW = 240, 430
def tile(img, title, sub, dim=False, badge=None):
    im = cv2.resize(img, (TW, TH), interpolation=cv2.INTER_AREA)
    if dim:
        im = (im.astype(np.float32) * 0.22).astype(np.uint8)
        (wf, hf), _ = cv2.getTextSize("FROZEN", cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3); cv2.putText(im, "FROZEN", ((TW - wf) // 2, (TH + hf) // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (90, 90, 240), 3)
    (w, h), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(im, (0, 0), (w + 12, h + 12), (0, 0, 0), -1); cv2.putText(im, title, (6, h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    if sub:
        (w2, h2), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(im, (0, h + 12), (w2 + 12, h + h2 + 24), (0, 0, 0), -1); cv2.putText(im, sub, (6, h + h2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 220, 250), 1)
    if badge:
        (w3, h3), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        col = (120, 230, 160) if badge[0] in "781" else ((255, 255, 255) if badge[0] in "n" else (150, 150, 255))
        cv2.rectangle(im, (TW - w3 - 14, TH - h3 - 14), (TW, TH), (0, 0, 0), -1); cv2.putText(im, badge, (TW - w3 - 7, TH - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    return im
P = [("Base (cloud model, 70 steps)", "bidirectional: whole clip at once, 0.078 fps", R + "/revisit/base_official_S05/video.mp4", "no stream"),
     ("Fast alone (on device, 4 steps)", "no cloud help", R + "/e2e55/control/video.mp4", "5/10"),
     ("Naive fix: polish Fast's own frames", "refine with Base low-noise expert, write back", R + "/resync55/W36_sink18/kvrw_naive_s1234/video.mp4", "3/10"),
     ("SYSTEM: cloud correction, pinned once", "refine prefix, Base regenerates 2.25 s, pin it", R + "/e2e55/fixed/video.mp4", "7/10"),
     ("Failure: anchor replaced every time", "same corrections, anchors overwritten", R + "/e2e55/rolling/video.mp4", "2/10"),
     ("Upper bound: Base anchors from t=0", "as if the cloud were there from the start", R + "/resync55/W36_sink18/resync_k6_s1234/video.mp4", "10/10")]
V = []
for name, sub, path, badge in P:
    fr = read(path); print(name, len(fr), flush=True); V.append((name, sub, fr, badge))
T = max(len(v[2]) for v in V)
wr = imageio.get_writer(O + "/23_base_fast_naive_system_55s.mp4", fps=16, quality=7, codec="libx264", pixelformat="yuv420p", macro_block_size=None, output_params=["-threads", "2"])
for t in range(T):
    cells = []
    for name, sub, fr, badge in V:
        over = t >= len(fr); f = fr[min(t, len(fr) - 1)]
        s2 = "cannot stream past 10 s" if (over and name.startswith("Base (")) else sub
        cells.append(tile(f, name, s2, dim=over, badge=badge))
    grid = np.concatenate([np.concatenate(cells[:3], 1), np.concatenate(cells[3:], 1)], 0)
    bar = np.zeros((36, grid.shape[1], 3), np.uint8)
    txt = "t = %5.2f s" % (t / 16); corr = any(abs(t / 16 - c) < 0.4 for c in (4.5, 19.5, 34.5))
    if corr: txt += "    <<< CLOUD CORRECTION"
    cv2.putText(bar, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (90, 255, 160) if corr else (255, 255, 255), 2)
    wr.append_data(np.concatenate([bar, grid], 0))
wr.close(); print("[SBS6_DONE]", T, flush=True)
