"""[10/16] Contact sheet of a clip for eyeballing: N evenly spaced frames (default 8) with their time stamps, 2 columns.
usage: /home/hk28965/specwm/scripts/envrun.sh python /home/hk28965/specwm/baselines/abot/frames_sheet.py VIDEO.mp4 OUT.jpg [--n 8 --fps 12 --w 480]"""
import sys, argparse, cv2, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("out"); ap.add_argument("--n", type=int, default=8); ap.add_argument("--fps", type=float, default=12.0); ap.add_argument("--w", type=int, default=480)
a = ap.parse_args(); cap = cv2.VideoCapture(a.video); fr = []
while True:
    ok, f = cap.read()
    if not ok: break
    fr.append(f)
idx = np.round(np.linspace(0, len(fr) - 1, a.n)).astype(int); H0, W0 = fr[0].shape[:2]; w = a.w; h = int(H0 * w / W0); cols = 2; rows = (a.n + 1) // 2
sheet = np.full((rows * (h + 20), cols * w, 3), 255, np.uint8)
for k, i in enumerate(idx):
    x = (k % cols) * w; y = (k // cols) * (h + 20); sheet[y + 20:y + 20 + h, x:x + w] = cv2.resize(fr[i], (w, h), interpolation=cv2.INTER_AREA)
    cv2.putText(sheet, "frame %d  t=%.2f s" % (i, i / a.fps), (x + 4, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
cv2.imwrite(a.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 85]); print(a.out, len(fr), "frames", sheet.shape)
