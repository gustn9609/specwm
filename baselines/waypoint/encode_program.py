"""[10/16 frozen baselines] Encode a PROTOCOL Section-2 control program (P1..P5) for Waypoint-1.5 — INLINE encoding written because
baselines/convert.py did not exist yet; it follows PROTOCOL Section 2 and the rules of tools/waypoint/poses_to_actions.py, but takes a pose array.
  program_to_c2w(pid, fps, n_frames) -> c2w (n_frames x 4 x 4), OpenCV camera (x right, y down, z forward), piecewise-constant per 2.5 s segment:
      yaw/pitch in deg/s integrated as R_{i+1} = R_i @ Ry(dyaw) @ Rx(dpitch) so that yaw = atan2(Rr[0,2], Rr[2,2]) and pitch = asin(-Rr[1,2])
      (Rr = R_i^T R_{i+1}) reproduce the program; translation in the camera frame (z forward) in u/s: t_{i+1} = t_i + R_i @ (0, 0, v/fps).
      NOTE on sign: with this (OpenCV) convention a positive yaw step moves the camera's forward axis towards +x (image right), i.e. the camera
      pans RIGHT and the scene content flows LEFT; PROTOCOL Section 2 labels positive yaw "left" — the formula, not the label, is used everywhere
      (the commanded c2w is the RPE reference, every converter derives its signs from the same c2w, so all models receive the same turn).
  encode_waypoint(c2w, k_yaw) -> one action per consecutive pose pair: {mouse: [k_yaw*dyaw_deg, k_yaw*dpitch_deg], buttons: [W|S|A|D keycodes]};
      a pose step = one Waypoint call (4 output frames at 60 fps -> 15 calls/s), so the program is sampled at fps=15 (150 calls for 10 s).
      Buttons (poses_to_actions rule): pressed when the camera-frame translation of the step exceeds 0.25 x median non-zero step speed;
      W if tr_z > |tr_x|, S if -tr_z > |tr_x|, else D (tr_x > 0) / A. Keycodes are Owl-Control / JS keycodes: W 87, S 83, A 65, D 68.
  k_yaw: --k-yaw, else baselines/calib.json ["waypoint"]["K_yaw"], else 118 (10/9 flow calibration, see ENV.md for why it must be re-measured).
usage: python encode_program.py PID OUT_DIR [--fps 15] [--seconds 10] [--k-yaw K]  -> OUT_DIR/actions.json, program_c2w.npy, program.json
[10/19 EXP 1] PID may be a long program of baselines/convert.py (P6, 60 s: --seconds defaults to its clip_s, 900 calls, encoded by convert.py only)
"""
import os, sys, json, argparse, numpy as np
W, S, A, D = 87, 83, 65, 68
CALIB = "/home/hk28965/specwm/baselines/calib.json"
# segments of 2.5 s: (forward speed u/s, yaw deg/s, pitch deg/s)   [PROTOCOL Section 2]
PROGRAMS = {
    "P1": [(1, 0, 0)] * 4,                                       # forward
    "P2": [(0, 18, 0)] * 4,                                      # turn: 180 deg pan
    "P3": [(1, 12, 0)] * 4,                                      # forward + turn
    "P4": [(1, 18, 0), (1, -18, 0), (1, -18, 0), (1, 18, 0)],    # S-curve (2 input flips)
    "P5": [(1, 0, 0), (0, 0, 0), (-1, 0, 0), (1, 0, 0)],         # stop-and-go: fwd, still, back, fwd
    "P2p": [(0, 16, 0)] * 4,                                     # 10/17: P2 / P4 at 16 deg/s (MG2's mouse cap saturates 18 deg/s); same rows as convert.PROGRAMS
    "P4p": [(1, 16, 0), (1, -16, 0), (1, -16, 0), (1, 16, 0)],
}
SEG_S = 2.5

def Ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def Rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def program_to_c2w(pid, fps, n_frames):
    seg = PROGRAMS[pid]; c2w = np.tile(np.eye(4), (n_frames, 1, 1))
    for i in range(n_frames - 1):
        v, yaw, pitch = seg[min(int((i / fps) / SEG_S), len(seg) - 1)]
        R = c2w[i, :3, :3]; c2w[i + 1, :3, :3] = R @ Ry(np.radians(yaw / fps)) @ Rx(np.radians(pitch / fps))
        c2w[i + 1, :3, 3] = c2w[i, :3, 3] + R @ np.array([0.0, 0.0, v / fps])
    return c2w

def relative_motion(c2w):
    """per step: (yaw_deg, pitch_deg, tr[3] in the camera frame of pose i) — the formulas of tools/waypoint/poses_to_actions.py"""
    R = c2w[:, :3, :3]; t = c2w[:, :3, 3]; mv = []
    for i in range(len(c2w) - 1):
        Rr = R[i].T @ R[i + 1]; yaw = np.degrees(np.arctan2(Rr[0, 2], Rr[2, 2])); pitch = np.degrees(np.arcsin(np.clip(-Rr[1, 2], -1, 1)))
        mv.append((float(yaw), float(pitch), R[i].T @ (t[i + 1] - t[i])))
    return mv

def encode_waypoint(c2w, k_yaw):
    mv = relative_motion(c2w); spd = np.array([np.linalg.norm(m[2]) for m in mv]); thr = 0.25 * np.median(spd[spd > 1e-9]) if (spd > 1e-9).any() else np.inf
    acts = []
    for yaw, pitch, tr in mv:
        b = []
        if np.linalg.norm(tr) > thr:
            if tr[2] > abs(tr[0]): b.append(W)
            elif -tr[2] > abs(tr[0]): b.append(S)
            elif tr[0] > 0: b.append(D)
            else: b.append(A)
        acts.append(dict(mouse=[float(k_yaw * yaw), float(k_yaw * pitch)], buttons=b))
    return acts

def default_k_yaw():
    if os.path.exists(CALIB):
        try:
            w = json.load(open(CALIB))["waypoint"]; return float(w["K_yaw"] if "K_yaw" in w else w["K_yaw_units_per_deg_per_call"]), "calib.json"
        except Exception: pass
    return 118.0, "default (10/9 flow calibration; unverified, see ENV.md)"

def encode_via_convert(pid, k, seconds):
    """the shared encoder (baselines/convert.py, appeared 10/16 22:11): c2w at 60 fps, 4 frames summed per call; identical to the inline encoder except
    at the two calls straddling a 2.5 s boundary in P4/P5 (convert.py averages them, inline switches at call 38/113). Returns (c2w, acts) or None."""
    try:
        sys.path.insert(0, "/home/hk28965/specwm/baselines"); import convert
        fps, fpc = 60, 4; c2w, _ = convert.program_to_c2w(pid, fps, int(round(fps * seconds)) + 1); acts = convert.encode_waypoint(c2w, k, fpc)
        assert len(acts) == int(round(fps * seconds / fpc)); return np.asarray(c2w), acts
    except Exception as e:
        print("[encode] convert.py not usable (%s: %s) -> inline encoder" % (type(e).__name__, e)); return None

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("pid"); ap.add_argument("out"); ap.add_argument("--fps", type=float, default=15.0, help="Waypoint call rate: 60 fps / 4 frames per call")
    ap.add_argument("--seconds", type=float, default=None, help="clip length; default 10, or [10/19] the program's clip_s for a long program of baselines/convert.py (P6: 60 s)"); ap.add_argument("--k-yaw", type=float, default=None); ap.add_argument("--inline", action="store_true", help="force the inline encoder even if convert.py exists"); a = ap.parse_args()
    SEGS = PROGRAMS.get(a.pid)   # [10/19 EXP 1, additive] long programs (P6) exist only in baselines/convert.py: segments + clip length come from there
    if a.seconds is None or SEGS is None:
        try:
            sys.path.insert(0, "/home/hk28965/specwm/baselines"); import convert as _cv
            if a.pid in getattr(_cv, "LONG_PROGRAMS", []):
                if a.seconds is None: a.seconds = _cv.clip_s_of(a.pid)
                if SEGS is None: SEGS = [list(x) for x in _cv.PROGRAMS[a.pid]["seg"]]
        except Exception as e: print("[encode] convert.py long-program lookup failed: %r" % e)
    if a.seconds is None: a.seconds = 10.0
    k, src = (a.k_yaw, "--k-yaw") if a.k_yaw is not None else default_k_yaw()
    n_calls = int(round(a.fps * a.seconds)); r = None if a.inline else encode_via_convert(a.pid, k, a.seconds)
    if r is not None: c2w, acts = r; enc = "baselines/convert.py (program_to_c2w @60 fps + encode_waypoint, K from calib.json)"
    else: c2w = program_to_c2w(a.pid, a.fps, n_calls + 1); acts = encode_waypoint(c2w, k); enc = "baselines/waypoint/encode_program.py (inline)"
    os.makedirs(a.out, exist_ok=True); np.save(a.out + "/program_c2w.npy", c2w.astype(np.float32)); json.dump(acts, open(a.out + "/actions.json", "w"))
    mv = relative_motion(c2w); tot_yaw = sum(m[0] for m in mv); tot_pitch = sum(m[1] for m in mv)
    meta = dict(program=a.pid, segments=SEGS, fps_calls=a.fps, seconds=a.seconds, calls=len(acts), c2w_poses=len(c2w), k_yaw=k, k_yaw_source=src, total_yaw_deg=tot_yaw, total_pitch_deg=tot_pitch,
                mouse_x_per_call_abs_max=max(abs(x["mouse"][0]) for x in acts), buttons_WSAD=[sum(key in x["buttons"] for x in acts) for key in (W, S, A, D)], encoder=enc)
    json.dump(meta, open(a.out + "/program.json", "w"), indent=1)
    print("[encode] %s: %d calls via %s, K_yaw %.1f (%s), total yaw %.1f deg, pitch %.1f deg, W/S/A/D calls %s, |mouse x| max %.1f" % (a.pid, len(acts), enc.split(" (")[0], k, src, tot_yaw, tot_pitch, meta["buttons_WSAD"], meta["mouse_x_per_call_abs_max"]))
