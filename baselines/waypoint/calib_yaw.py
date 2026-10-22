"""[10/16 frozen baselines] Realised camera yaw of Waypoint clips, recovered with Depth Anything 3 (same model/source as scripts/rpe_da3.py), for the
K_yaw calibration demanded by PROTOCOL rule 3 (calibrate on the BASELINE's own outputs). Per video: sample every --stride frames, run DA3 on windows of
--win sampled frames (consecutive windows share one frame), take the recovered w2c extrinsics and compute the per-step yaw/pitch in the camera frame with
the SAME formulas as the encoder (R = c2w rotation = w2c[:3,:3]^T, Rr = R_i^T R_{i+1}, yaw = atan2(Rr[0,2], Rr[2,2]), pitch = asin(-Rr[1,2])). Relative
rotations are invariant to DA3's global Sim(3) ambiguity, so no alignment is needed. Cumulative yaw / calls = realised deg per call; with the mouse units
per call that produced the clip this gives K_yaw = units / deg. A CPU phase-correlation horizontal shift (px/frame; scene content moving LEFT = camera
panning RIGHT in image terms) is printed as a direction cross-check.
usage: CUDA_VISIBLE_DEVICES=g envrun.sh python calib_yaw.py OUT.json LABEL=VIDEO:MOUSE_X_PER_CALL ... [--stride 4 --win 16 --res 504 --frames-per-call 4]"""
import sys, os, json, argparse, numpy as np, torch, cv2
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("specs", nargs="+"); ap.add_argument("--stride", type=int, default=4); ap.add_argument("--win", type=int, default=16)
ap.add_argument("--res", type=int, default=504); ap.add_argument("--frames-per-call", type=int, default=4); ap.add_argument("--model", default="/home/hk28965/specwm/hf/DA3NESTED-GIANT-LARGE-1.1")
ap.add_argument("--device", default="cuda"); ap.add_argument("--max-frames", type=int, default=0, help="debug: only the first N frames"); a = ap.parse_args()
sys.path.insert(0, "/home/hk28965/specwm/tools/mg35/Matrix-Game-3.5/third_party/depth-anything-3/src")
from depth_anything_3.api import DepthAnything3
def frames_of(path):
    cap = cv2.VideoCapture(path); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return fr
def yaw_pitch(w2c_a, w2c_b):
    Rr = w2c_a[:3, :3] @ w2c_b[:3, :3].T   # R_a^T R_b with R = w2c^T
    return float(np.degrees(np.arctan2(Rr[0, 2], Rr[2, 2]))), float(np.degrees(np.arcsin(np.clip(-Rr[1, 2], -1, 1))))
def phase_shift(fr):
    g = [cv2.cvtColor(cv2.resize(f, (512, 256)), cv2.COLOR_RGB2GRAY).astype(np.float32) for f in fr]
    return float(np.mean([cv2.phaseCorrelate(g[i], g[i + 1])[0][0] for i in range(len(g) - 1)]) * fr[0].shape[1] / 512)
OUT = json.load(open(a.out)) if os.path.exists(a.out) else {}; model = None
for spec in a.specs:
    label, rest = spec.split("=", 1); path, mouse = (rest.split(":") + ["nan"])[:2]; mouse = float(mouse)
    if model is None: model = DepthAnything3.from_pretrained(a.model).to(a.device).eval()
    fr = frames_of(path); fr = fr[:a.max_frames] if a.max_frames else fr; idx = list(range(0, len(fr), a.stride)); calls = len(fr) / a.frames_per_call
    yaws, pitches, fxs, k = [], [], [], 0
    while k < len(idx) - 1:
        ii = idx[k:k + a.win]
        if len(ii) < 2: break
        with torch.no_grad(): pred = model.inference([fr[i] for i in ii], process_res=a.res)
        ex = np.asarray(pred.extrinsics, dtype=np.float64)
        for j in range(len(ii) - 1): y, p = yaw_pitch(ex[j], ex[j + 1]); yaws.append(y); pitches.append(p)
        try: fxs += list(np.asarray(pred.intrinsics)[:, 0, 0] * fr[0].shape[1] / float(pred.processed_width if hasattr(pred, "processed_width") else a.res))
        except Exception: pass
        k += a.win - 1
    tot = float(np.sum(yaws)); per_call = tot / calls; shift = phase_shift(fr); fx = float(np.median(fxs)) if fxs else float("nan")
    flow_yaw_per_call = float(np.degrees(np.arctan2(-shift * a.frames_per_call, fx))) if fxs else float("nan")   # content shift left (negative) = camera yaw right (positive)
    r = dict(label=label, video=path, frames=len(fr), calls=calls, mouse_x_per_call=mouse, yaw_total_deg=round(tot, 2), yaw_deg_per_call=round(per_call, 4), yaw_deg_per_s=round(per_call * 60 / a.frames_per_call, 2),
             pitch_total_deg=round(float(np.sum(pitches)), 2), yaw_steps_deg=[round(y, 3) for y in yaws], K_yaw_units_per_deg=(round(mouse / per_call, 1) if abs(per_call) > 1e-6 and not np.isnan(mouse) else None),
             phase_shift_px_per_frame=round(shift, 3), direction="camera pans RIGHT (content flows left)" if shift < -0.3 else ("camera pans LEFT (content flows right)" if shift > 0.3 else "no clear pan"),
             da3_fx_px_at_full_width=round(fx, 1), flow_yaw_deg_per_call=round(flow_yaw_per_call, 4), flow_yaw_total_deg=round(flow_yaw_per_call * calls, 2),
             K_yaw_from_flow=(round(mouse / flow_yaw_per_call, 1) if abs(flow_yaw_per_call) > 1e-6 and not np.isnan(mouse) else None))
    OUT[label] = r; json.dump(OUT, open(a.out, "w"), indent=1)
    print("[calib] %-14s frames %4d calls %5.1f mouse %7.1f/call -> yaw %7.2f deg total = %6.3f deg/call (%5.1f deg/s), pitch %6.2f, K_yaw %s, shift %.2f px/frame (%s)" % (label, len(fr), calls, mouse, tot, per_call, per_call * 60 / a.frames_per_call, r["pitch_total_deg"], r["K_yaw_units_per_deg"], shift, r["direction"]), flush=True)
