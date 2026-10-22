#!/usr/bin/env python3
"""[10/17 frozen baselines] Write the NOT-YET-DONE subset of a baseline job list, so a queue chunk is idempotent and never pays a model load / warm-up
for nothing.  Input = a JSON list of jobs with an "out" dir (abot / mg2 / astra schema) or a Waypoint LIST.tsv (IMAGE  P  SEED  NAME; out = --out-root/NAME).
done = <out>/video.mp4 AND <out>/timing.json exist, and timing.json does not say frame_check_ok == false (ABot's layout check).
Prints the number of pending jobs on stdout (the chunk script reads it); exit 0 normally, 2 on a bad input.  --redo keeps every job.  Plain python3.
usage: python3 filter_pending.py JOBS.json|LIST.tsv PENDING.json|PENDING.tsv [--out-root DIR] [--redo]"""
import os, sys, json, argparse

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--out-root", default=None, help="TSV only: runs dir that holds <NAME>/")
ap.add_argument("--redo", action="store_true", help="keep every job (regenerate)"); a = ap.parse_args()


def is_done(out):
    v, t = os.path.join(out, "video.mp4"), os.path.join(out, "timing.json")
    if not (os.path.isfile(v) and os.path.isfile(t)):
        return False
    try:
        T = json.load(open(t))
    except Exception:
        return False                                          # unreadable timing.json -> redo the clip
    return T.get("frame_check_ok", True) is not False


if not os.path.isfile(a.src):
    print("filter_pending: no such file %s" % a.src, file=sys.stderr); sys.exit(2)
os.makedirs(os.path.dirname(os.path.abspath(a.dst)) or ".", exist_ok=True)
if a.src.endswith(".tsv"):
    if not a.out_root:
        print("filter_pending: --out-root is required for a TSV list", file=sys.stderr); sys.exit(2)
    keep, n_all = [], 0
    for line in open(a.src):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        f = [x.strip() for x in s.split("\t")]
        if len(f) < 4:
            print("filter_pending: TSV line needs IMAGE P SEED NAME: %r" % s, file=sys.stderr); sys.exit(2)
        n_all += 1
        if a.redo or not is_done(os.path.join(a.out_root, f[3])):
            keep.append(line.rstrip("\n"))
    with open(a.dst, "w") as fh:
        fh.write("# pending subset of %s (%d of %d)\n" % (a.src, len(keep), n_all))
        for l in keep: fh.write(l + "\n")
else:
    jobs = json.load(open(a.src))
    if not isinstance(jobs, list):
        print("filter_pending: %s is not a JSON list" % a.src, file=sys.stderr); sys.exit(2)
    n_all = len(jobs); keep = [j for j in jobs if a.redo or not is_done(j["out"])]
    json.dump(keep, open(a.dst, "w"), indent=1)
print(len(keep))
