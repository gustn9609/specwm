"""Merge a DA3 mouse calibration (calib_da3.py --out) into baselines/calib.json["mg2"] without touching other models' entries.
Adds: K_per_deg_per_frame_da3, largest_unsaturated_yaw_deg_per_s, measurement {...}, n_frames_generated / n_frames_written; sets K_per_deg_per_frame
to the DA3 value only with --apply (otherwise the 10/9 value stays and the new one is recorded next to it).
usage: /home/hk28965/specwm/scripts/envrun.sh python /home/hk28965/specwm/baselines/mg2/calib_merge.py CALIB_RESULT.json [--apply]"""
import sys, json, time
B = "/home/hk28965/specwm/baselines"; res = json.load(open(sys.argv[1])); apply = "--apply" in sys.argv
C = json.load(open(B + "/calib.json")); m = C.setdefault("mg2", {})
m["K_per_deg_per_frame_da3"] = round(res["K_mouse_per_deg_frame"], 4); m["largest_unsaturated_yaw_deg_per_s"] = round(res["largest_unsaturated_yaw_deg_per_s"], 2)
m["n_frames_generated"] = 129; m["n_frames_written"] = 120; m["n_frames_note"] = "MG2 needs frames % 4 == 1 and latents % 3 == 0: 121 frames = 31 latents is not generable; 129 frames (33 latents) are generated and the first 120 (10.0 s) written"
m["measurement_da3"] = dict(date=time.strftime("%Y-%m-%d %H:%M %Z"), video=res["video"], program=res["program"], commanded_mouse_yaw=res["commanded_mouse_yaw_median"], commanded_deg_per_s=res["commanded_deg_per_s"],
                             realized_deg_per_frame=round(res["realized_deg_per_frame_da3"], 4), realized_deg_per_s=round(res["realized_deg_per_s_da3"], 2), realized_total_deg=round(res["realized_total_deg_da3"], 1), per_segment_deg_per_frame=[round(x, 4) for x in res["realized_per_segment_deg_per_frame"]],
                             sign_ok=res["sign_ok"], flow_median_px_per_frame=round(res["flow_median_px_per_frame"], 3), da3_model=res["da3_model"], method=res["method"], program_saturated_at_K_0067=res["program_saturated"])
if apply: m["K_per_deg_per_frame_10_09"] = m.get("K_per_deg_per_frame", 0.067); m["K_per_deg_per_frame"] = m["K_per_deg_per_frame_da3"]; m["source"] = m.get("source", "") + " | K re-calibrated on MG2's own P2 output with DA3 (baselines/mg2/calib_da3.py, see measurement_da3)"
json.dump(C, open(B + "/calib.json", "w"), indent=1); print("calib.json mg2 updated: K_da3 %.4f (applied=%s), largest unsaturated %.1f deg/s" % (m["K_per_deg_per_frame_da3"], apply, m["largest_unsaturated_yaw_deg_per_s"]))
