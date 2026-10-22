"""Merge the Waypoint calibration into the shared baselines/calib.json (only the "waypoint" key is touched; other models' entries are kept).
usage: python write_calib.py K_YAW "source text" [--measurements calib_da3.json] [--extra key=value ...]"""
import sys, os, json, argparse, datetime
ap = argparse.ArgumentParser(); ap.add_argument("k_yaw", type=float); ap.add_argument("source"); ap.add_argument("--measurements", default=""); ap.add_argument("--extra", nargs="*", default=[])
ap.add_argument("--path", default="/home/hk28965/specwm/baselines/calib.json"); a = ap.parse_args()
C = json.load(open(a.path)) if os.path.exists(a.path) else {}
entry = dict(K_yaw=a.k_yaw, unit="mouse-x units per degree of yaw per 4-frame call (15 calls/s); mouse = [K_yaw*dyaw_deg, K_yaw*dpitch_deg]; positive mouse x = camera pans right",
             source=a.source, date=datetime.datetime.now().isoformat(timespec="minutes"), measured_on="baselines/waypoint/smoke (highland_castle_loch, P2, seed 1234) + constant-mouse sweep; DA3 pose recovery (calib_yaw.py)")
if a.measurements and os.path.exists(a.measurements):
    M = json.load(open(a.measurements)); entry["measurements"] = {k: {f: v[f] for f in ("mouse_x_per_call", "calls", "yaw_total_deg", "yaw_deg_per_call", "K_yaw_units_per_deg", "phase_shift_px_per_frame", "da3_fx_px_at_full_width", "flow_yaw_deg_per_call", "K_yaw_from_flow") if f in v} for k, v in M.items()}
for kv in a.extra: k, v = kv.split("=", 1); entry[k] = v
old = C.get("waypoint", {}) if isinstance(C.get("waypoint"), dict) else {}
if "source" in old and "source_previous" not in old: old["source_previous"] = old["source"]
old.update(entry); old["K_yaw_units_per_deg_per_call"] = a.k_yaw   # keep other agents' fields; both key names carry the same value
C["waypoint"] = old; json.dump(C, open(a.path, "w"), indent=1, ensure_ascii=False); print("wrote", a.path, "waypoint.K_yaw =", a.k_yaw, "keys", list(old.keys()))
