"""Frames of a clip side by side for a visual check: python contact_sheet.py VIDEO.mp4 OUT.png [--idx 0,24,48,72,96,119] [--scale 0.5]"""
import sys, argparse, cv2, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("out"); ap.add_argument("--idx", default="0,24,48,72,96,119"); ap.add_argument("--scale", type=float, default=0.5); a = ap.parse_args()
cap = cv2.VideoCapture(a.video); fr = []
while True:
    ok, f = cap.read()
    if not ok: break
    fr.append(f)
idx = [min(int(i), len(fr) - 1) for i in a.idx.split(",")]; tiles = []
for i in idx:
    f = cv2.resize(fr[i], None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA); cv2.putText(f, "f%d" % i, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3); cv2.putText(f, "f%d" % i, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1); tiles.append(f)
n = len(tiles); cols = 3 if n > 4 else n; rows = (n + cols - 1) // cols; h, w = tiles[0].shape[:2]; sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
for k, t in enumerate(tiles): r, c = divmod(k, cols); sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
cv2.imwrite(a.out, sheet); print(a.out, "frames", len(fr), "sheet", sheet.shape)
