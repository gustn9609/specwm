#!/usr/bin/env python3
"""[10/19 audit] Compare the FROZEN Waypoint clips (runs/waypoint/<clip>/) with the _idle timing re-measure (runs/waypoint_idle/<clip>/, gen/queue/8/300_*):
per clip md5 of video.mp4 frozen vs idle (Waypoint determinism is unknown: this REPORTS it) and steady_fps_gen / gen_s / e2e_s / ttff_s of both.
Writes runs/waypoint_idle/COMPARE.json = {n_frozen, n_idle, n_compared, identical (count), identical_list, differing (list), missing_idle (list),
fps: {clip: {frozen, idle}}, median_fps_frozen, median_fps_idle}.  Plain python3 (hashlib + json), no GPU.  Idempotent.
usage: python3 -I /home/hk28965/specwm/baselines/gen/waypoint_idle_compare.py [--frozen DIR] [--idle DIR]"""
import os, sys, json, glob, hashlib, argparse
B = "/home/hk28965/specwm/baselines"
ap = argparse.ArgumentParser(); ap.add_argument("--frozen", default=B + "/runs/waypoint"); ap.add_argument("--idle", default=B + "/runs/waypoint_idle"); a = ap.parse_args()
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def num(x):
    try: return float(x)
    except Exception: return None
def med(v): v = sorted(x for x in v if x is not None); return v[len(v) // 2] if v else None
clips = sorted(os.path.basename(d.rstrip("/")) for d in glob.glob(a.frozen + "/*/") if "_long" not in d and os.path.isfile(d + "timing.json"))
ident, diff, miss, fps = [], [], [], {}
for c in clips:
    fv, iv = "%s/%s/video.mp4" % (a.frozen, c), "%s/%s/video.mp4" % (a.idle, c); it = "%s/%s/timing.json" % (a.idle, c)
    if not (os.path.isfile(iv) and os.path.isfile(it)): miss.append(c); continue
    (ident if md5(fv) == md5(iv) else diff).append(c)
    F, I = json.load(open("%s/%s/timing.json" % (a.frozen, c))), json.load(open(it))
    fps[c] = {k: dict(frozen=num(F.get(k)), idle=num(I.get(k))) for k in ("steady_fps_gen", "steady_fps_e2e", "gen_s", "e2e_s", "ttff_s")}
    fps[c]["idle_caveat"] = I.get("timing_caveat")
out = dict(n_frozen=len(clips), n_idle=len(ident) + len(diff), n_compared=len(ident) + len(diff), identical=len(ident), identical_list=ident, differing=diff, missing_idle=miss, fps=fps,
           median_fps_frozen=med([v["steady_fps_gen"]["frozen"] for v in fps.values()]), median_fps_idle=med([v["steady_fps_gen"]["idle"] for v in fps.values()]),
           note="[10/19 audit] md5 of video.mp4 frozen (runs/waypoint, measured on a shared GPU, timing_caveat) vs idle (runs/waypoint_idle, exclusively held GPU, same image / program / seed 1234); identical = bitwise same mp4. Waypoint determinism was unknown before this run: a non-zero 'differing' count means the re-run is a timing-only measurement, not a reproduction of the frozen video.")
os.makedirs(a.idle, exist_ok=True); json.dump(out, open(a.idle + "/COMPARE.json", "w"), indent=1)
print("[waypoint_idle_compare] frozen %d, idle %d, identical %d, differing %d, missing %d, median fps_gen frozen %s idle %s -> %s/COMPARE.json" % (len(clips), out["n_idle"], len(ident), len(diff), len(miss), out["median_fps_frozen"], out["median_fps_idle"], a.idle))
