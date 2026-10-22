"""[10/16 frozen baselines] Calibrate Matrix-Game 2.0's mouse unit on ITS OWN output (PROTOCOL 1.3): recover the camera path of a generated clip with
Depth Anything 3 (same model/code path as scripts/rpe_da3.py), take the per-frame yaw with the converters' formula (yaw = atan2(Rr[0,2], Rr[2,2]),
Rr = R_i^T R_{i+1} on c2w; + = pan right) and compare it with the commanded mouse yaw of the clip -> K = mouse units per degree per frame, the
largest yaw rate (deg/s at the clip's fps) that stays below their 0.1 cap, and whether the commanded program saturated. A CPU optical-flow proxy
(cv2 DIS, median horizontal flow px/frame; negative = content moves left = pan right) is printed as a sign/magnitude cross-check.
usage: CUDA_VISIBLE_DEVICES=g /home/hk28965/specwm/scripts/envrun.sh python /home/hk28965/specwm/baselines/mg2/calib_da3.py VIDEO.mp4 TIMING.json [--stride 1 --res 504 --out calib_mg2.json]"""
import sys, os, json, argparse, numpy as np, cv2, torch
S = "/home/hk28965/specwm"; sys.path.insert(0, S + "/tools/mg35/Matrix-Game-3.5/third_party/depth-anything-3/src")
ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("timing"); ap.add_argument("--stride", type=int, default=1); ap.add_argument("--res", type=int, default=504)
ap.add_argument("--model", default=S + "/hf/DA3NESTED-GIANT-LARGE-1.1"); ap.add_argument("--out", default=""); ap.add_argument("--max-frames", type=int, default=120); a = ap.parse_args()
T = json.load(open(a.timing)); fps = T["fps_native"]; cap = cv2.VideoCapture(a.video); fr = []
while True:
    ok, f = cap.read()
    if not ok or len(fr) >= a.max_frames: break
    fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
H, W = fr[0].shape[:2]; print("[calib] %d frames %dx%d, fps %d, program %s" % (len(fr), W, H, fps, T["program"]), flush=True)
# commanded mouse yaw per frame (their units) for this clip
sys.path.insert(0, S + "/baselines"); from convert import program_to_c2w, encode_mg2, frame_motion as relative_motion
c2w_cmd, _ = program_to_c2w(T["program"], fps, T["frames_generated"]); enc = encode_mg2(c2w_cmd, T["control"]["K_mouse_per_deg_frame"], T["control"]["mouse_cap"])
mouse_yaw = np.array(enc["mouse"], np.float64)[1:len(fr), 1]; cmd_deg = np.array([m[0] for m in relative_motion(c2w_cmd[:len(fr)])])
print("[calib] commanded: mouse yaw median %.4f (cap %.2f, unclipped %.4f), program yaw %.3f deg/frame = %.1f deg/s" % (np.median(mouse_yaw), T["control"]["mouse_cap"], T["control"]["max_abs_mouse_unclipped"], np.median(cmd_deg), np.median(cmd_deg) * fps), flush=True)
# optical-flow proxy (CPU)
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); flows = []
for i in range(len(fr) - 1):
    g0, g1 = cv2.cvtColor(fr[i], cv2.COLOR_RGB2GRAY), cv2.cvtColor(fr[i + 1], cv2.COLOR_RGB2GRAY); fl = dis.calc(g0, g1, None); flows.append(float(np.median(fl[..., 0])))
flows = np.array(flows); print("[calib] flow: median horizontal %.2f px/frame (negative = pan right), per-2.5s-segment medians %s" % (np.median(flows), [round(float(np.median(flows[s * int(2.5 * fps):(s + 1) * int(2.5 * fps)])), 2) for s in range(4)]), flush=True)
# DA3 pose recovery
from depth_anything_3.api import DepthAnything3
model = DepthAnything3.from_pretrained(a.model).to("cuda").eval(); idx = list(range(0, len(fr), a.stride)); t0 = torch.cuda.Event(enable_timing=True); torch.cuda.reset_peak_memory_stats()
with torch.no_grad(): pred = model.inference([fr[i] for i in idx], process_res=a.res)
ext = np.asarray(pred.extrinsics, np.float64); intr = np.asarray(pred.intrinsics, np.float64) if getattr(pred, "intrinsics", None) is not None else None
w2c = np.tile(np.eye(4), (len(ext), 1, 1)); w2c[:, :3, :4] = ext[:, :3, :4]; c2w = np.linalg.inv(w2c)
mv = relative_motion(c2w); yaw = np.array([m[0] for m in mv]) / a.stride; pitch = np.array([m[1] for m in mv]) / a.stride   # deg per original frame
seg = int(2.5 * fps / a.stride); per_seg = [float(np.median(yaw[s * seg:(s + 1) * seg])) for s in range(4) if len(yaw[s * seg:(s + 1) * seg])]
yaw_med = float(np.median(yaw)); total = float(yaw.sum() * a.stride)
fx = float(np.median(intr[:, 0, 0])) if intr is not None else float("nan"); sx = (W / pred.processed_width) if hasattr(pred, "processed_width") else float("nan")
print("[calib] DA3: %d frames, per-frame yaw median %.3f deg (segments %s), total %.1f deg over %.1f s, pitch median %.3f; DA3 fx %.1f px (at its process res); peak %.1f GiB" % (len(idx), yaw_med, [round(x, 3) for x in per_seg], total, len(fr) / fps, float(np.median(pitch)), fx, torch.cuda.max_memory_allocated() / 2 ** 30), flush=True)
m_cmd = float(np.median(np.abs(mouse_yaw))); deg_per_frame = abs(yaw_med); K = m_cmd / deg_per_frame if deg_per_frame > 1e-6 else float("nan")
max_unsat_dps = T["control"]["mouse_cap"] / K * fps if K == K else float("nan")
res = dict(video=a.video, program=T["program"], fps=fps, frames=len(fr), commanded_mouse_yaw_median=m_cmd, commanded_deg_per_frame=float(np.median(cmd_deg)), commanded_deg_per_s=float(np.median(cmd_deg) * fps),
           realized_deg_per_frame_da3=yaw_med, realized_deg_per_s_da3=yaw_med * fps, realized_total_deg_da3=total, realized_per_segment_deg_per_frame=per_seg, sign_ok=bool(np.sign(yaw_med) == np.sign(np.median(cmd_deg))) if abs(np.median(cmd_deg)) > 0 else None,
           flow_median_px_per_frame=float(np.median(flows)), da3_fx_px=fx, K_mouse_per_deg_frame=K, largest_unsaturated_yaw_deg_per_s=max_unsat_dps, program_saturated=bool(T["control"]["saturated"]), da3_model=a.model, stride=a.stride, res=a.res,
           method="DA3 (DA3NESTED-GIANT-LARGE-1.1) poses of the MG2 clip -> per-frame yaw via atan2(Rr[0,2],Rr[2,2]); K = |commanded mouse yaw| / |realized deg per frame|; largest unsaturated rate = cap / K * fps")
print("[calib] K = %.4f mouse units per deg per frame; realized %.1f deg/s for commanded mouse %.3f; largest unsaturated yaw rate = %.1f deg/s (cap %.2f at %d fps); sign_ok=%s" % (K, yaw_med * fps, m_cmd, max_unsat_dps, T["control"]["mouse_cap"], fps, res["sign_ok"]), flush=True)
if a.out: json.dump(res, open(a.out, "w"), indent=1); print("[calib] wrote", a.out)
