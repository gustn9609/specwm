"""[10/17 frozen baselines] Realised camera motion of a generated clip, recovered with Depth Anything 3 (DA3NESTED-GIANT-LARGE-1.1, the same
model/source as scripts/rpe_da3.py and baselines/waypoint/calib_yaw.py), for the Astra translation/rotation calibration demanded by PROTOCOL
rule 3 (calibrate magnitudes on the BASELINE's own outputs). The nested DA3 is METRIC: its second branch estimates metric depth, the poses'
translations are scaled to it (depth_anything_3/model/da3.py NestedDepthAnything3Net._apply_depth_alignment: extrinsics[:, :, :3, 3] *= scale_factor,
is_metric = 1), so per-step translations come out in ~metres, one scale per DA3 window.

Per video: sample every --stride frames, run DA3 on windows of --win sampled frames (consecutive windows share one frame), take the recovered
w2c extrinsics and compute the per-step motion in the camera frame with the converters' formulas (c2w = inv(w2c), Rr = R_i^T R_{i+1},
yaw = atan2(Rr[0,2], Rr[2,2]) (+ = pan right), pitch = asin(-Rr[1,2]), tr = R_i^T (t_{i+1} - t_i), tr[2] = forward). Relative rotations are
invariant to DA3's global Sim(3) ambiguity; relative translations carry each window's metric scale (reported per window, with its scale_factor).
Commanded motion comes from the clip's camera.npz (astra_gen.py: c2w at the clip's fps + the (L,13) Astra rows) or from a poses.npy (ours).
Cross-checks on the CPU: phase-correlation horizontal shift (content LEFT = camera pans RIGHT) and a DIS optical-flow divergence proxy
(median radial flow outward > 0 = moving forward).

usage: CUDA_VISIBLE_DEVICES=g /home/hk28965/specwm/scripts/envrun.sh python /home/hk28965/specwm/baselines/astra/calib_da3.py OUT.json LABEL=VIDEO.mp4:CAMERA.npz[:FPS] ...
       (CAMERA = camera.npz from astra_gen.py [keys c2w, astra] or a poses.npy [c2w only]; FPS default 20 for camera.npz, 16 for poses.npy)
       [--stride 2 --win 16 --res 504 --max-frames 0]"""
import sys, os, json, argparse, importlib.util, numpy as np, torch, cv2
S = "/home/hk28965/specwm"; B = S + "/baselines"
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("specs", nargs="+"); ap.add_argument("--stride", type=int, default=2); ap.add_argument("--win", type=int, default=16)
ap.add_argument("--res", type=int, default=504); ap.add_argument("--model", default=S + "/hf/DA3NESTED-GIANT-LARGE-1.1"); ap.add_argument("--device", default="cuda")
ap.add_argument("--max-frames", type=int, default=0, help="debug: only the first N frames"); ap.add_argument("--seg-s", type=float, default=2.5); a = ap.parse_args()
sys.path.insert(0, S + "/tools/mg35/Matrix-Game-3.5/third_party/depth-anything-3/src")
from depth_anything_3.api import DepthAnything3
_sp = importlib.util.spec_from_file_location("convert", B + "/convert.py"); convert = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(convert)
S_FLIP = np.diag([1.0, -1.0, -1.0])


def frames_of(path):
    cap = cv2.VideoCapture(path); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return fr


def to44(ex):
    ex = np.asarray(ex, dtype=np.float64)
    if ex.shape[-2:] == (4, 4): return ex
    out = np.tile(np.eye(4), (len(ex), 1, 1)); out[:, :3, :4] = ex[:, :3, :4]; return out


def step_motion(w2c_a, w2c_b):
    """camera-frame motion a -> b from two w2c: (yaw_deg, pitch_deg, tr[3]) with the converters' formulas on c2w = inv(w2c)"""
    A, Bm = np.linalg.inv(w2c_a), np.linalg.inv(w2c_b); Ra, Rb = A[:3, :3], Bm[:3, :3]; Rr = Ra.T @ Rb
    return (float(np.degrees(np.arctan2(Rr[0, 2], Rr[2, 2]))), float(np.degrees(np.arcsin(np.clip(-Rr[1, 2], -1, 1)))), Ra.T @ (Bm[:3, 3] - A[:3, 3]))


def phase_shift(fr):
    g = [cv2.cvtColor(cv2.resize(f, (512, 256)), cv2.COLOR_RGB2GRAY).astype(np.float32) for f in fr]
    return float(np.mean([cv2.phaseCorrelate(g[i], g[i + 1])[0][0] for i in range(len(g) - 1)]) * fr[0].shape[1] / 512)


def flow_divergence(fr, step=2):
    """median radial component of the DIS flow (px/frame at 512x256 scaled to full width) from the image centre: > 0 = expansion = forward motion"""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); vals = []
    h, w = 256, 512; yy, xx = np.mgrid[0:h, 0:w]; cx, cy = w / 2.0, h / 2.0; rx, ry = xx - cx, yy - cy; rn = np.sqrt(rx ** 2 + ry ** 2) + 1e-6; ux, uy = rx / rn, ry / rn
    ring = (rn > 0.25 * h) & (rn < 0.9 * h)
    for i in range(0, len(fr) - step, step):
        g0 = cv2.cvtColor(cv2.resize(fr[i], (w, h)), cv2.COLOR_RGB2GRAY); g1 = cv2.cvtColor(cv2.resize(fr[i + step], (w, h)), cv2.COLOR_RGB2GRAY)
        fl = dis.calc(g0, g1, None); rad = (fl[..., 0] * ux + fl[..., 1] * uy)[ring]; vals.append(float(np.median(rad)) / step)
    return float(np.median(vals)) * fr[0].shape[1] / w, [round(v * fr[0].shape[1] / w, 3) for v in vals]


def commanded_of(cam_path, fps, n_frames):
    """(per-frame yaw deg, per-frame forward u, astra rows or None) for the first n_frames of the commanded path"""
    if cam_path.endswith(".npz"):
        z = np.load(cam_path); c2w = z["c2w"].astype(np.float64); astra = z["astra"] if "astra" in z.files else None
    else:
        c2w = np.load(cam_path).astype(np.float64); astra = None
    c2w = c2w[:n_frames]; mv = convert.frame_motion(c2w)
    yaw = np.array([m[0] for m in mv]); fwd = np.array([m[2][2] for m in mv]); path = np.array([np.linalg.norm(m[2]) for m in mv])
    out = dict(frames=len(c2w), yaw_deg_per_s=float(yaw.sum() / ((len(c2w) - 1) / fps)), fwd_u_per_s=float(fwd.sum() / ((len(c2w) - 1) / fps)), path_u=float(path.sum()), yaw_total_deg=float(yaw.sum()))
    if astra is not None:
        L = (len(c2w) - 1) // 4; rows = astra[1:L + 1]                   # rows 1..L = the generated latents (row 0 = condition latent)
        ya, tz, tn = [], [], []
        for r in rows:
            M = r[:12].reshape(3, 4); Rr = S_FLIP @ M[:, :3] @ S_FLIP; tr = S_FLIP @ M[:, 3]
            ya.append(np.degrees(np.arctan2(Rr[0, 2], Rr[2, 2]))); tz.append(tr[2]); tn.append(np.linalg.norm(tr))
        out["astra_rows"] = int(L); out["astra_yaw_deg_per_latent"] = float(np.mean(ya)); out["astra_fwd_units_per_latent"] = float(np.mean(tz)); out["astra_path_units_per_latent"] = float(np.mean(tn))
        out["astra_yaw_total_deg"] = float(np.sum(ya)); out["astra_path_units_total"] = float(np.sum(tn))
    return out


OUT = json.load(open(a.out)) if os.path.exists(a.out) else {}; model = None
for spec in a.specs:
    label, rest = spec.split("=", 1); parts = rest.split(":"); video, cam = parts[0], parts[1]; fps = float(parts[2]) if len(parts) > 2 else (20.0 if cam.endswith(".npz") else 16.0)
    if model is None: model = DepthAnything3.from_pretrained(a.model).to(a.device).eval()
    fr = frames_of(video); fr = fr[:a.max_frames] if a.max_frames else fr; N = len(fr); idx = list(range(0, N, a.stride)); dur = (N - 1) / fps
    yaws, pitches, trs, step_frames, wins, k = [], [], [], [], [], 0
    torch.cuda.reset_peak_memory_stats()
    while k < len(idx) - 1:
        ii = idx[k:k + a.win]
        if len(ii) < 2: break
        with torch.no_grad(): pred = model.inference([fr[i] for i in ii], process_res=a.res)
        ex = to44(pred.extrinsics); sf = float(pred.scale_factor) if getattr(pred, "scale_factor", None) is not None else float("nan"); met = int(getattr(pred, "is_metric", 0) or 0)
        wy, wz, wn = [], [], []
        for j in range(len(ii) - 1):
            y, p, tr = step_motion(ex[j], ex[j + 1]); yaws.append(y); pitches.append(p); trs.append(tr); step_frames.append(ii[j + 1] - ii[j]); wy.append(y); wz.append(tr[2]); wn.append(np.linalg.norm(tr))
        nfr = ii[-1] - ii[0]
        wins.append(dict(frames=[ii[0], ii[-1]], is_metric=met, scale_factor=round(sf, 4), yaw_deg_per_s=round(float(np.sum(wy)) * fps / nfr, 3), fwd_m_per_s=round(float(np.sum(wz)) * fps / nfr, 4), path_m_per_s=round(float(np.sum(wn)) * fps / nfr, 4),
                         da3_fx_px=round(float(np.median(np.asarray(pred.intrinsics)[:, 0, 0])) * fr[0].shape[1] / float(getattr(pred, "processed_width", a.res)), 1) if getattr(pred, "intrinsics", None) is not None else None))
        k += a.win - 1
    yaws = np.array(yaws); pitches = np.array(pitches); trs = np.array(trs); stepf = np.array(step_frames, dtype=np.float64); fwd = trs[:, 2]; pathn = np.linalg.norm(trs, axis=1)
    covered_s = stepf.sum() / fps
    yaw_ps, pitch_ps, fwd_ps, path_ps = float(yaws.sum() / covered_s), float(pitches.sum() / covered_s), float(fwd.sum() / covered_s), float(pathn.sum() / covered_s)
    # per 2.5 s segment (by the step's start frame)
    starts = np.cumsum(np.concatenate([[0], stepf[:-1]])); segs = []
    for s in range(int(np.ceil(dur / a.seg_s))):
        m = (starts / fps >= s * a.seg_s) & (starts / fps < (s + 1) * a.seg_s)
        if m.sum() < 2: continue
        t = stepf[m].sum() / fps; segs.append(dict(seg=s, yaw_deg_per_s=round(float(yaws[m].sum() / t), 3), fwd_m_per_s=round(float(fwd[m].sum() / t), 4), path_m_per_s=round(float(pathn[m].sum() / t), 4), lateral_m_per_s=round(float(trs[m, 0].sum() / t), 4), vertical_m_per_s=round(float(trs[m, 1].sum() / t), 4)))
    shift = phase_shift(fr); div, div_steps = flow_divergence(fr)
    cmd = commanded_of(cam, fps, N) if cam and cam != "-" else {}
    lat_s = 4.0 / fps                                                     # seconds per latent (Wan VAE stride 4)
    r = dict(label=label, video=video, camera=cam, fps=fps, frames=N, duration_s=round(dur, 3), stride=a.stride, win=a.win, res=a.res, da3_model=a.model, da3_is_metric=int(all(w["is_metric"] for w in wins)),
             realised=dict(yaw_deg_per_s=round(yaw_ps, 3), yaw_total_deg=round(float(yaws.sum()), 2), yaw_deg_per_latent=round(yaw_ps * lat_s, 4), pitch_deg_per_s=round(pitch_ps, 3),
                           fwd_m_per_s=round(fwd_ps, 4), fwd_m_per_latent=round(fwd_ps * lat_s, 5), path_m_per_s=round(path_ps, 4), path_m_per_latent=round(path_ps * lat_s, 5), path_m_total=round(float(pathn.sum()), 4), fwd_m_total=round(float(fwd.sum()), 4),
                           lateral_m_per_s=round(float(trs[:, 0].sum() / covered_s), 4), vertical_m_per_s=round(float(trs[:, 1].sum() / covered_s), 4),
                           m_per_deg=(round(fwd_ps / yaw_ps, 5) if abs(yaw_ps) > 0.5 else None), fwd_m_per_s_window_median=round(float(np.median([w["fwd_m_per_s"] for w in wins])), 4), fwd_m_per_s_window_iqr=[round(float(np.percentile([w["fwd_m_per_s"] for w in wins], q)), 4) for q in (25, 75)],
                           scale_factor_window_median=round(float(np.nanmedian([w["scale_factor"] for w in wins])), 4), per_segment=segs, windows=wins),
             commanded=cmd, crosscheck=dict(phase_shift_px_per_frame=round(shift, 3), pan_direction="camera pans RIGHT (content flows left)" if shift < -0.3 else ("camera pans LEFT (content flows right)" if shift > 0.3 else "no clear pan"),
                                            flow_divergence_px_per_frame=round(div, 3), flow_direction="forward (expansion)" if div > 0.15 else ("backward (contraction)" if div < -0.15 else "no clear zoom"), flow_divergence_steps=div_steps),
             peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), yaw_steps_deg=[round(float(y), 3) for y in yaws], fwd_steps_m=[round(float(z), 4) for z in fwd])
    if cmd:
        g = {}
        if abs(cmd["yaw_deg_per_s"]) > 0.5: g["yaw_gain"] = round(yaw_ps / cmd["yaw_deg_per_s"], 4)
        if abs(cmd["fwd_u_per_s"]) > 1e-6: g["m_per_u"] = round(fwd_ps / cmd["fwd_u_per_s"], 5)
        if "astra_fwd_units_per_latent" in cmd and abs(cmd["astra_fwd_units_per_latent"]) > 1e-9: g["m_per_astra_unit"] = round(fwd_ps * lat_s / cmd["astra_fwd_units_per_latent"], 4)
        if "astra_yaw_deg_per_latent" in cmd and abs(cmd["astra_yaw_deg_per_latent"]) > 1e-3: g["astra_yaw_gain"] = round(yaw_ps * lat_s / cmd["astra_yaw_deg_per_latent"], 4)
        if abs(cmd["yaw_deg_per_s"]) > 0.5 and abs(cmd["fwd_u_per_s"]) > 1e-6: g["commanded_u_per_deg"] = round(cmd["fwd_u_per_s"] / cmd["yaw_deg_per_s"], 5)
        r["gains"] = g
    OUT[label] = r; json.dump(OUT, open(a.out, "w"), indent=1)
    print("[calib] %-18s %d fr @%g fps | realised yaw %7.2f deg (%6.2f deg/s) pitch %6.2f | fwd %7.3f m (%.3f m/s = %.4f m/latent, windows median %.3f IQR %s) path %.3f m | metric %d scale~%.3f | cmd yaw %.2f deg/s fwd %.3f u/s%s | shift %.2f px (%s) div %.2f (%s) | gains %s | peak %.1f GiB" % (
        label, N, fps, yaws.sum(), yaw_ps, pitches.sum(), fwd.sum(), fwd_ps, fwd_ps * lat_s, r["realised"]["fwd_m_per_s_window_median"], r["realised"]["fwd_m_per_s_window_iqr"], pathn.sum(), r["da3_is_metric"], r["realised"]["scale_factor_window_median"],
        cmd.get("yaw_deg_per_s", float("nan")), cmd.get("fwd_u_per_s", float("nan")), (" (astra rows: %.3f deg, %.4f units per latent)" % (cmd["astra_yaw_deg_per_latent"], cmd["astra_fwd_units_per_latent"]) if "astra_rows" in cmd else ""),
        shift, r["crosscheck"]["pan_direction"], div, r["crosscheck"]["flow_direction"], json.dumps(r.get("gains", {})), r["peak_alloc_GiB"]), flush=True)
