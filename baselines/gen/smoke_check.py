#!/usr/bin/env python3
"""[10/19 tier-5 smokes] Verify one smoke clip in the frozen layout and write the gate sentinel that the tier-6 chunks of that path wait for:
  baselines/gen/sentinels/<NAME>.ok   (JSON: what was checked + key numbers)   or   <NAME>.fail (JSON: why)      exit 0 = ok, 1 = fail.
Checks (every clip id given): <runs>/<id>/video.mp4 + timing.json exist, the video has exactly --frames frames (ffprobe -count_frames; ffprobe from
~/.local/bin or tools/ffmpeg-7.0.2-amd64-static when present), timing.json frames == --frames, frame_check_ok is not false (ABot), and per --kind:
  ours    frames_generated >= --frames (our loader generated the whole program), steps / gen_s / steady_fps_gen present
  hyb     the same for BOTH streams <runs_dev>/<id> and <runs_cloud>/<id>; the device received every correction (corrections_received == the chunk
          count) and applied >= 1 (KV feedback works), the cloud refined every chunk; --ee: the device timing has mean_steps in (1, 4)
  base    nothing more (the runner's own timing.json)
--rc: the generation's exit status (non-zero -> fail without looking further). Plain python3.
usage: python3 smoke_check.py NAME --kind ours|hyb|base --runs DIR [--runs-cloud DIR] --ids a,b --frames 961 [--rc N] [--ee] [--note TEXT]"""
import os, sys, json, glob, time, argparse, subprocess
B = "/home/hk28965/specwm/baselines"; SENT = B + "/gen/sentinels"
ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--kind", required=True, choices=["ours", "hyb", "base"]); ap.add_argument("--runs", required=True)
ap.add_argument("--runs-cloud", default=""); ap.add_argument("--ids", required=True); ap.add_argument("--frames", type=int, required=True); ap.add_argument("--rc", type=int, default=0)
ap.add_argument("--ee", action="store_true"); ap.add_argument("--note", default=""); a = ap.parse_args()
os.makedirs(SENT, exist_ok=True)
FFPROBE = next((p for p in ("/home/hk28965/.local/bin/ffprobe", "/home/hk28965/specwm/tools/ffmpeg-7.0.2-amd64-static/ffprobe") if os.path.exists(p)), "ffprobe")
def nframes(v):
    o = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames,width,height,r_frame_rate", "-of", "json", v], capture_output=True, text=True)
    s = json.loads(o.stdout)["streams"][0]; return int(s["nb_read_frames"]), "%sx%s" % (s["width"], s["height"]), s["r_frame_rate"]
problems, info = [], dict(name=a.name, kind=a.kind, frames_expected=a.frames, rc=a.rc, note=a.note, checked=time.strftime("%Y-%m-%d %H:%M:%S"), clips={})
if a.rc != 0: problems.append("generation exit status %d" % a.rc)
dirs = [("", a.runs)] + ([("cloud", a.runs_cloud)] if a.kind == "hyb" else [])
for cid in [x for x in a.ids.split(",") if x]:
    for tag, rd in dirs:
        d = os.path.join(rd, cid); v, t = d + "/video.mp4", d + "/timing.json"; key = cid + ("/" + tag if tag else "")
        if not (os.path.isfile(v) and os.path.isfile(t)): problems.append("%s: missing video.mp4 / timing.json in %s" % (key, d)); continue
        try: T = json.load(open(t))
        except Exception as e: problems.append("%s: unreadable timing.json (%r)" % (key, e)); continue
        try: n, res, fr = nframes(v)
        except Exception as e: problems.append("%s: ffprobe failed (%r)" % (key, e)); continue
        rec = dict(dir=d, video_frames=n, resolution=res, rate=fr, frames=T.get("frames"), frames_generated=T.get("frames_generated"), steps=T.get("steps"), gen_s=T.get("gen_s"),
                   e2e_s=T.get("e2e_s"), steady_fps_gen=T.get("steady_fps_gen"), steady_fps_e2e=T.get("steady_fps_e2e"), peak_alloc_GiB=T.get("peak_alloc_GiB"), ttff_s=T.get("ttff_s"))
        if n != a.frames: problems.append("%s: video has %d frames, expected %d" % (key, n, a.frames))
        if T.get("frames") not in (None, a.frames): problems.append("%s: timing.json frames %s != %d" % (key, T.get("frames"), a.frames))
        if T.get("frame_check_ok") is False: problems.append("%s: frame_check_ok false (%s)" % (key, T.get("frame_check")))
        if a.kind in ("ours", "hyb"):
            if not (isinstance(T.get("frames_generated"), int) and T["frames_generated"] >= a.frames): problems.append("%s: frames_generated %s < %d" % (key, T.get("frames_generated"), a.frames))
            if T.get("gen_s") is None or T.get("steady_fps_gen") is None: problems.append("%s: gen_s / steady_fps_gen missing" % key)
        if a.kind == "hyb" and not tag:   # device stream: KV feedback must have happened
            rec.update(corrections_received=T.get("corrections_received"), corrections_applied=T.get("corrections_applied"), chunks_sent=T.get("chunks_sent"), cloud_refined=T.get("cloud_refined"),
                       corr_lag_median_s=T.get("corr_lag_median_s"), device_fps_shown=T.get("device_fps_shown"), mean_steps=T.get("mean_steps"))
            nch = len(T.get("chunk_s") or [])
            if not (T.get("corrections_received") == nch and (T.get("corrections_applied") or 0) >= 1): problems.append("%s: corrections received %s / applied %s for %d chunks" % (key, T.get("corrections_received"), T.get("corrections_applied"), nch))
            if T.get("cloud_refined") != nch: problems.append("%s: cloud refined %s of %d chunks" % (key, T.get("cloud_refined"), nch))
            if a.ee and not (isinstance(T.get("mean_steps"), (int, float)) and 1.0 < T["mean_steps"] < 4.0): problems.append("%s: early-exit mean_steps %s not in (1, 4)" % (key, T.get("mean_steps")))
        info["clips"][key] = rec
info["ok"] = not problems; info["problems"] = problems
for f in (SENT + "/" + a.name + ".ok", SENT + "/" + a.name + ".fail"):
    if os.path.exists(f): os.remove(f)
out = SENT + "/" + a.name + (".ok" if not problems else ".fail"); tmp = out + ".tmp"; json.dump(info, open(tmp, "w"), indent=1); os.replace(tmp, out)
print("[smoke_check] %s: %s -> %s" % (a.name, "OK" if not problems else "FAIL: " + "; ".join(problems), out))
sys.exit(0 if not problems else 1)
