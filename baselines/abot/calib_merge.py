"""[10/17] Merge DA3 yaw measurements of ABot's own clips (baselines/abot/calib_da3.py OUT.json) into baselines/calib.json["abot"] WITHOUT
rewriting any other key: adds/updates "abot"."measurements"[label] (one entry per measured clip), the summary keys L_key_yaw_deg_per_s_da3,
L_pan_direction, P2_realised_over_commanded, block_layout_note, measured_on, date (all under "abot"), nothing else. The primary label (--primary,
default = the first) feeds the summary keys. Re-running overwrites only the labels present in OUT.json.
usage: /home/hk28965/specwm/scripts/envrun.sh python /home/hk28965/specwm/baselines/abot/calib_merge.py OUT.json [--primary LABEL] [--calib /home/hk28965/specwm/baselines/calib.json]"""
import sys, json, time, argparse, os
ap = argparse.ArgumentParser(); ap.add_argument("result"); ap.add_argument("--primary", default=None); ap.add_argument("--calib", default="/home/hk28965/specwm/baselines/calib.json"); a = ap.parse_args()
R = json.load(open(a.result)); assert R, "empty result"; prim = a.primary or next(iter(R)); assert prim in R, prim
C = json.load(open(a.calib)); ab = C.setdefault("abot", {}); M = ab.setdefault("measurements", {})
for label, r in R.items():
    M[label] = dict(video=r["video"], program=r["program"], keys_held="".join(r.get("keys_held", [])), fps=r["fps"], frames_used=r["frames_used"], seconds=r["seconds"],
                    yaw_total_deg=r["yaw_total_deg"], yaw_deg_per_frame=r["yaw_deg_per_frame"], yaw_deg_per_s=r["yaw_deg_per_s"], yaw_deg_by_second=r["yaw_deg_by_second"],
                    pitch_total_deg=r["pitch_total_deg"], commanded=r["commanded"], realised_over_commanded=r["realised_over_commanded"], sign_ok=r["sign_ok"], direction=r["direction"],
                    phase_shift_px_per_frame=r["phase_shift_px_per_frame"], flow_direction=r["flow_direction"], flow_yaw_deg_per_s=r["flow_yaw_deg_per_s"], da3_fx_px_at_full_width=r["da3_fx_px_at_full_width"],
                    da3_model=r["da3_model"], stride=r["stride"], win=r["win"], res=r["res"], method=r["method"], date=r["date"])
p = R[prim]
ab["L_key_yaw_deg_per_s_da3"] = p["yaw_deg_per_s"]; ab["L_pan_direction"] = p["direction"]; ab["P2_realised_over_commanded"] = p["realised_over_commanded"]
ab["block_layout_note"] = "run.sh/bench_abot.py use ABot's real layout: block 0 = 9 decoded frames (start image + 2 latent frames x 4), then 12 per block; 11 blocks = 129 generated, first 120 (10.0 s) written; the streaming VAE is reset per stream (10/17 fix), block 0 = 9 frames asserted"
ab["measured_on"] = "DA3 pose recovery (baselines/abot/calib_da3.py) of ABot's own P2 clip(s) %s; primary = %s" % (", ".join(sorted(R)), prim); ab["date"] = time.strftime("%Y-%m-%dT%H:%M")
json.dump(C, open(a.calib, "w"), indent=1)
print("calib.json abot updated: measurements %s; primary %s: L key = %.2f deg/s (%s), realised/commanded %s" % (sorted(M), prim, p["yaw_deg_per_s"], p["direction"], p["realised_over_commanded"]))
