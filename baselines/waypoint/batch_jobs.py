"""[10/17 frozen baselines] Prepare a Waypoint batch: LIST.tsv (IMAGE  PROGRAM_ID  SEED  [OUTNAME|-]  [K_YAW]) -> OUTROOT/<name>/{actions.json, program_c2w.npy,
program.json} via encode_program.py (K_yaw from baselines/calib.json) and OUTROOT/batch_jobs.json for run_waypoint_program.py --batch.
usage: envrun.sh python batch_jobs.py LIST.tsv OUTROOT [--k-yaw K]"""
import os, sys, json, argparse, subprocess
B = os.path.dirname(os.path.abspath(__file__)); PY = sys.executable
ap = argparse.ArgumentParser(); ap.add_argument("list"); ap.add_argument("outroot"); ap.add_argument("--k-yaw", type=float, default=None)
ap.add_argument("--jobs-out", default=None, help="[10/17] where to write the jobs json (default OUTROOT/batch_jobs.json)"); a = ap.parse_args()
jobs, seen = [], set()
for ln, line in enumerate(open(a.list), 1):
    line = line.strip()
    if not line or line.startswith("#"): continue
    f = [x.strip() for x in line.split("\t")]
    if len(f) < 3: f = line.split()                       # tolerate whitespace-separated lines without spaces in paths
    if len(f) < 3: sys.exit("LIST line %d: need IMAGE PROGRAM_ID SEED [OUTNAME]: %r" % (ln, line))
    img, pid, seed = os.path.abspath(f[0]), f[1], int(f[2])
    name = f[3] if len(f) > 3 and f[3] not in ("", "-") else "%s__%s__%s" % (os.path.basename(os.path.dirname(img)), os.path.splitext(os.path.basename(img))[0], pid)
    k = float(f[4]) if len(f) > 4 and f[4] not in ("", "-") else a.k_yaw   # optional 5th column: per-clip K_yaw override (calibration runs only; frozen runs use calib.json)
    if not os.path.exists(img): sys.exit("LIST line %d: image not found: %s" % (ln, img))
    if name in seen: sys.exit("LIST line %d: duplicate clip name %s" % (ln, name))
    seen.add(name); out = os.path.join(os.path.abspath(a.outroot), name); os.makedirs(out, exist_ok=True)
    cmd = [PY, os.path.join(B, "encode_program.py"), pid, out] + (["--k-yaw", str(k)] if k is not None else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode: sys.exit("encode failed for %s: %s" % (name, r.stdout[-800:] + r.stderr[-800:]))
    print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "[encode] %s" % name)
    jobs.append(dict(image=img, actions=os.path.join(out, "actions.json"), out=out, seed=seed, program=pid, name=name, k_yaw=k))
jp = os.path.abspath(a.jobs_out) if a.jobs_out else os.path.join(os.path.abspath(a.outroot), "batch_jobs.json"); os.makedirs(os.path.dirname(jp), exist_ok=True); json.dump(jobs, open(jp, "w"), indent=1)
print("[batch_jobs] %d clips -> %s" % (len(jobs), jp))
