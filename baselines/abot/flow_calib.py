"""[10/16 frozen baselines] CPU-only, provisional pan calibration of an ABot clip from optical flow (the PROTOCOL asks for DA3 pose recovery —
scripts/rpe_da3.py, GPU — for the frozen number; this is the quick check used right after a smoke run).
Per frame pair: Farneback flow on a 4x downscaled grey frame, median horizontal flow u (px/frame at full res) over the central 60 % of the image;
the camera yaw per frame is approx. -atan(u / fx) with fx = 0.9 * W (PROTOCOL Section 2 intrinsics): image content moving LEFT (u < 0) means the
camera pans to its RIGHT (+yaw by the converters' formula, see baselines/calib.json yaw_pitch_convention). Prints per-second yaw rate, the total,
and the sign; writes <video>.flow.json next to the clip unless --out is given.
usage: /home/hk28965/specwm/scripts/envrun.sh python /home/hk28965/specwm/baselines/abot/flow_calib.py VIDEO.mp4 [--fps 12] [--out X.json]"""
import sys, json, argparse, numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("--fps", type=float, default=12.0); ap.add_argument("--out", default=""); ap.add_argument("--fx-frac", type=float, default=0.9)
a = ap.parse_args(); cap = cv2.VideoCapture(a.video); fr = []
while True:
    ok, f = cap.read()
    if not ok: break
    fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
H, W = fr[0].shape; s = 4; sm = [cv2.resize(f, (W // s, H // s), interpolation=cv2.INTER_AREA) for f in fr]; fx = a.fx_frac * W
u, v = [], []
for i in range(len(sm) - 1):
    fl = cv2.calcOpticalFlowFarneback(sm[i], sm[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0); h, w = fl.shape[:2]
    c = fl[int(0.2 * h):int(0.8 * h), int(0.2 * w):int(0.8 * w)]; u.append(float(np.median(c[..., 0])) * s); v.append(float(np.median(c[..., 1])) * s)
u = np.array(u); v = np.array(v); yaw = -np.degrees(np.arctan(u / fx)); pitch = np.degrees(np.arctan(v / fx))
per_s = [float(yaw[int(k * a.fps):int((k + 1) * a.fps)].sum()) for k in range(int(len(yaw) / a.fps))]
res = dict(video=a.video, frames=len(fr), size=[W, H], fx=fx, yaw_total_deg=float(yaw.sum()), yaw_deg_per_s_mean=float(yaw.sum() / (len(yaw) / a.fps)), yaw_deg_per_s_by_second=per_s,
           pitch_total_deg=float(pitch.sum()), median_u_px_per_frame=float(np.median(u)), direction="camera pans RIGHT (+yaw by the converters' formula; content moves left)" if yaw.sum() > 0 else "camera pans LEFT (-yaw; content moves right)",
           method="Farneback flow, 4x downscale, central 60 %, median; yaw = -atan(u/fx), fx = %.2f*W; provisional, not DA3" % a.fx_frac)
print(json.dumps(res, indent=1)); json.dump(res, open(a.out or a.video + ".flow.json", "w"), indent=1)
