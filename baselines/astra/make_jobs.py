"""[10/17 frozen baselines] Write a JOBS.json for `baselines/astra/run.sh batch` from a scene set x programs (PROTOCOL §3/§4 layout):
out = baselines/runs/astra<suffix>/<set>__<scene>__<P>/ ; prompt = the scene's .txt (official / our caption) unless --no-caption (empty prompt arm).
usage: python make_jobs.py SET [--programs P1,P2,P3,P4,P5] [--seed 1234] [--no-caption] [--suffix _nocap] [--out JOBS.json]   (plain python3 is enough)"""
import argparse, glob, json, os
B = "/home/hk28965/specwm/baselines"
ap = argparse.ArgumentParser(); ap.add_argument("set"); ap.add_argument("--programs", default="P1,P2,P3,P4,P5"); ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--no-caption", action="store_true"); ap.add_argument("--suffix", default="", help="runs/astra<suffix>/ and task-id suffix"); ap.add_argument("--out", default="")
ap.add_argument("--scenes", default="", help="comma list of scene names to keep (default all)")
ap.add_argument("--prefix", default="", help="[10/17] out-dir / job 'set' prefix (default = SET): neutral_832 for the neutral scenes at Astra's 832x480 (ours set neutral_832)")
ap.add_argument("--scenes-dir", default="", help="[10/17] scene image dir (default baselines/scenes/<SET>)")
a = ap.parse_args()
sd = a.scenes_dir or "%s/scenes/%s" % (B, a.set); prefix = a.prefix or a.set
scenes = sorted(glob.glob(sd + "/*.png") + glob.glob(sd + "/*.jpg")); assert scenes, "no scenes in %s" % sd
keep = set(a.scenes.split(",")) if a.scenes else None; jobs = []
if keep: missing = keep - {os.path.splitext(os.path.basename(s))[0] for s in scenes}; assert not missing, "--scenes not in %s: %s" % (sd, sorted(missing))
for img in scenes:
    name = os.path.splitext(os.path.basename(img))[0]
    if keep and name not in keep: continue
    t = os.path.splitext(img)[0] + ".txt"; prompt = "" if a.no_caption or not os.path.exists(t) else open(t).read()
    for P in a.programs.split(","):
        jobs.append(dict(image=img, program=P, seed=a.seed, out="%s/runs/astra%s/%s__%s__%s%s" % (B, a.suffix, prefix, name, P, a.suffix), prompt=prompt, set=prefix, scene=name))
out = a.out or "%s/runs/astra%s/jobs_%s%s.json" % (B, a.suffix, prefix, a.suffix); os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(jobs, open(out, "w"), indent=1); print(out, len(jobs), "jobs (%d scenes x %s)" % (len({j["scene"] for j in jobs}), a.programs))
