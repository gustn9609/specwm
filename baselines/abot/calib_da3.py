"""[10/17 frozen baselines] Realised camera yaw of ABot-World clips, recovered with Depth Anything 3 (same model/source as scripts/rpe_da3.py and
baselines/waypoint/calib_yaw.py), for PROTOCOL rule 3 (calibrate on the BASELINE's own outputs). ABot has no magnitude input — a held J/L key is its
fixed pan speed — so the calibration is the realised yaw rate of the L key (deg/s and total over the 10 s P2 clip) and its physical direction.
Per run dir (video.mp4 + timing.json): the first --max-frames frames (120 = 10 s) sampled every --stride frames, DA3 on windows of --win sampled
frames (consecutive windows share one frame, so the relative rotations chain across windows), per-step yaw/pitch in the camera frame with the
SAME formulas as the encoder (R = c2w rotation = w2c[:3,:3]^T, Rr = R_i^T R_{i+1}, yaw = atan2(Rr[0,2], Rr[2,2]), pitch = asin(-Rr[1,2]); relative
rotations are invariant to DA3's global Sim(3) ambiguity, no alignment needed). The commanded yaw of the program over the same frames comes from
encode_abot.program_to_c2w. A CPU phase-correlation horizontal shift (px/frame; content moving LEFT = camera panning RIGHT) is the direction cross-check.
usage: CUDA_VISIBLE_DEVICES=g /home/hk28965/specwm/scripts/envrun.sh python /home/hk28965/specwm/baselines/abot/calib_da3.py OUT.json LABEL=RUNDIR ...
       [--stride 2 --win 16 --res 504 --max-frames 120]"""
import sys, os, json, argparse, time, numpy as np, torch, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/hk28965/specwm/tools/mg35/Matrix-Game-3.5/third_party/depth-anything-3/src")
from encode_abot import program_to_c2w, relative_motions, KEYS
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("specs", nargs="+", help="LABEL=RUNDIR (dir with video.mp4 + timing.json) or LABEL=VIDEO.mp4:PROGRAM")
ap.add_argument("--stride", type=int, default=2); ap.add_argument("--win", type=int, default=16); ap.add_argument("--res", type=int, default=504); ap.add_argument("--max-frames", type=int, default=120)
ap.add_argument("--model", default="/home/hk28965/specwm/hf/DA3NESTED-GIANT-LARGE-1.1"); ap.add_argument("--device", default="cuda"); a = ap.parse_args()
from depth_anything_3.api import DepthAnything3
def frames_of(path, n):
    cap = cv2.VideoCapture(path); fr = []
    while True:
        ok, f = cap.read()
        if not ok or (n and len(fr) >= n): break
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
    label, rest = spec.split("=", 1)
    if os.path.isdir(rest):
        T = json.load(open(os.path.join(rest, "timing.json"))); video = os.path.join(rest, T.get("video", "video.mp4")); program = T.get("program"); fps = T.get("fps_native", 12)
        keys = [r["keys"] for r in T["rows"]]
    else:
        video, program = (rest.split(":") + [None])[:2]; T = {}; fps = 12; keys = []
    if model is None: model = DepthAnything3.from_pretrained(a.model).to(a.device).eval(); torch.cuda.reset_peak_memory_stats()
    fr = frames_of(video, a.max_frames); idx = list(range(0, len(fr), a.stride)); yaws, pitches, fxs, k = [], [], [], 0; t0 = time.time()
    while k < len(idx) - 1:
        ii = idx[k:k + a.win]
        if len(ii) < 2: break
        with torch.no_grad(): pred = model.inference([fr[i] for i in ii], process_res=a.res)
        ex = np.asarray(pred.extrinsics, dtype=np.float64)
        for j in range(len(ii) - 1): y, p = yaw_pitch(ex[j], ex[j + 1]); yaws.append(y); pitches.append(p)
        try: fxs += list(np.asarray(pred.intrinsics)[:, 0, 0] * fr[0].shape[1] / float(pred.processed_width if hasattr(pred, "processed_width") else a.res))
        except Exception: pass
        k += a.win - 1
    n_steps = len(yaws); frames_covered = idx[min(k, len(idx) - 1)] + 1 if idx else 0; secs = (frames_covered - 1) / fps
    tot = float(np.sum(yaws)); per_frame = tot / max(frames_covered - 1, 1); per_s = per_frame * fps
    steps_per_s = fps / a.stride; by_second = [round(float(np.sum(yaws[int(s * steps_per_s):int((s + 1) * steps_per_s)])), 2) for s in range(int(n_steps / steps_per_s))]
    shift = phase_shift(fr[:frames_covered]); fx = float(np.median(fxs)) if fxs else float("nan")
    flow_yaw_per_frame = float(np.degrees(np.arctan2(-shift, fx))) if fxs else float("nan")   # content shift left (negative px) = camera yaw right (positive)
    cmd = None
    if program:
        c2w = program_to_c2w(program, fps, frames_covered); mv = relative_motions(c2w); cmd = dict(program=program, yaw_total_deg=round(float(sum(m[0] for m in mv)), 2), yaw_deg_per_s=round(float(sum(m[0] for m in mv)) / secs, 3) if secs > 0 else None, pitch_total_deg=round(float(sum(m[1] for m in mv)), 2))
    held = sorted({c for ks in keys for c in ks}) if keys else []
    r = dict(label=label, video=video, program=program, keys_per_block=keys, keys_held=held, fps=fps, frames_used=frames_covered, seconds=round(secs, 3), stride=a.stride, win=a.win, res=a.res, n_steps=n_steps,
             yaw_total_deg=round(tot, 2), yaw_deg_per_frame=round(per_frame, 4), yaw_deg_per_s=round(per_s, 2), yaw_deg_by_second=by_second, yaw_steps_deg=[round(y, 3) for y in yaws],
             pitch_total_deg=round(float(np.sum(pitches)), 2), pitch_deg_per_s=round(float(np.sum(pitches)) / secs, 3) if secs > 0 else None,
             commanded=cmd, realised_over_commanded=(round(tot / cmd["yaw_total_deg"], 4) if cmd and abs(cmd["yaw_total_deg"]) > 1e-6 else None),
             sign_ok=(bool(np.sign(tot) == np.sign(cmd["yaw_total_deg"])) if cmd and abs(cmd["yaw_total_deg"]) > 1e-6 else None),
             direction="camera pans RIGHT (+yaw by the converters' formula; content flows left)" if tot > 0 else "camera pans LEFT (-yaw; content flows right)",
             phase_shift_px_per_frame=round(shift, 3), flow_direction="content flows left -> camera pans RIGHT" if shift < -0.3 else ("content flows right -> camera pans LEFT" if shift > 0.3 else "no clear pan"),
             da3_fx_px_at_full_width=round(fx, 1), flow_yaw_deg_per_frame=round(flow_yaw_per_frame, 4), flow_yaw_deg_per_s=round(flow_yaw_per_frame * fps, 2),
             da3_model=a.model, da3_s=round(time.time() - t0, 1), da3_peak_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2), date=time.strftime("%Y-%m-%d %H:%M %Z"),
             method="DA3 (DA3NESTED-GIANT-LARGE-1.1) w2c poses on windows of %d frames sampled every %d (shared boundary frame); per-step yaw = atan2(Rr[0,2],Rr[2,2]), Rr = R_i^T R_{i+1}, R = w2c^T; summed over the first %d frames" % (a.win, a.stride, frames_covered))
    OUT[label] = r; json.dump(OUT, open(a.out, "w"), indent=1)
    print("[calib] %-28s keys %s frames %3d (%.2f s) -> yaw %7.2f deg total = %6.3f deg/frame = %6.2f deg/s (commanded %s), pitch %6.2f, by second %s; shift %.2f px/frame (%s); DA3 %.0f s peak %.1f GiB" % (
        label, "".join(held) or "-", frames_covered, secs, tot, per_frame, per_s, "%s %.1f deg = %.1f deg/s" % (program, cmd["yaw_total_deg"], cmd["yaw_deg_per_s"]) if cmd else "n/a", r["pitch_total_deg"], by_second, shift, r["flow_direction"], r["da3_s"], r["da3_peak_GiB"]), flush=True)
print("[calib] wrote", a.out)
