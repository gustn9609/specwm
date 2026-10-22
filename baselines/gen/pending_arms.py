#!/usr/bin/env python3
"""[10/17 frozen baselines] Which runs/<model_arm>/ dirs still have clips without metrics?  Prints one line per arm:  <arm>\t<n_pending>\t<n_ready>
A clip is READY when <clip>/video.mp4 + timing.json exist, its name parses as <set>__<scene>__P<id>[suffix] and video.mp4 is older than --min-age s
(a generator may still be writing it).  It is PENDING when any of these is missing under label == arm (score_runs.sh's rule: an entry counts only
with that label): judge  metrics/vlm.json  ('quality' or 'drift');  vbench metrics/vbench6.json ('quality_score');  rpe metrics/rpe.json ('rot');
reaction metrics/reaction.json ('switches'), *__P5* clips only.  --verbose lists the pending clips with the missing metric names.  Plain python3.
usage: python3 pending_arms.py [--runs DIR] [--metrics DIR] [--min-age 120] [--arms a,b] [--verbose]"""
import os, re, sys, json, glob, time, argparse
B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--runs", default=B + "/runs"); ap.add_argument("--metrics", default=B + "/metrics")
ap.add_argument("--min-age", type=float, default=120.0, help="seconds since video.mp4 was last modified"); ap.add_argument("--arms", default="", help="comma list; default every dir under --runs")
ap.add_argument("--verbose", action="store_true"); a = ap.parse_args()


def load(name):
    p = os.path.join(a.metrics, name)
    try:
        d = json.load(open(p)) if os.path.exists(p) else {}
    except Exception as e:
        print("pending_arms: unreadable %s (%s)" % (p, e), file=sys.stderr); d = {}
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and k.startswith("/"): out[k] = v; out[os.path.realpath(k)] = v
    return out


def scored(m, vid, label, keys):
    r = m.get(vid) or m.get(os.path.abspath(vid)) or m.get(os.path.realpath(vid))
    return isinstance(r, dict) and r.get("label") == label and any(k in r for k in keys)


M = dict(judge=(load("vlm.json"), ("quality", "drift")), vbench=(load("vbench6.json"), ("quality_score",)), rpe=(load("rpe.json"), ("rot",)), reaction=(load("reaction.json"), ("switches",)))
now = time.time(); arms = [x for x in a.arms.split(",") if x] or sorted(os.path.basename(d) for d in glob.glob(os.path.join(a.runs, "*")) if os.path.isdir(d))
for arm in arms:
    n_ready, pend = 0, []
    for cd in sorted(glob.glob(os.path.join(a.runs, arm, "*"))):
        name = os.path.basename(cd); parts = name.split("__")
        if not os.path.isdir(cd) or len(parts) < 3 or not re.match(r"P\d+", parts[-1]): continue
        vid, tj = os.path.join(cd, "video.mp4"), os.path.join(cd, "timing.json")
        if not (os.path.isfile(vid) and os.path.isfile(tj)): continue
        if now - os.path.getmtime(vid) < a.min_age: continue
        n_ready += 1; missing = [k for k, (m, keys) in M.items() if (k != "reaction" or parts[-1].startswith("P5")) and not scored(m, vid, arm, keys)]
        if missing: pend.append((name, missing))
    if pend:
        print("%s\t%d\t%d" % (arm, len(pend), n_ready))
        if a.verbose:
            for name, missing in pend: print("\t%s\t%s" % (name, ",".join(missing)))
