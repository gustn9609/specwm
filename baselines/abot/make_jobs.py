"""[10/17 frozen baselines] Write a batch job list for baselines/abot/bench_abot.py --batch (ONE 5B model load for many clips):
one job per (scene image, program) with out = OUT_ROOT/<set>__<scene>__<program> (the runs/ layout of PROTOCOL Section 4), seed fixed, prompt =
the image's sibling .txt (empty if none, or forced empty with --no-caption). Plain python (no numpy).
usage: python3 /home/hk28965/specwm/baselines/abot/make_jobs.py JOBS.json --out-root /home/hk28965/specwm/baselines/runs/abot [--set abot]
       [--scenes-dir /home/hk28965/specwm/baselines/scenes/abot | --images IMG ...] [--programs P1,P2,P3,P4,P5] [--seed 1234] [--no-caption] [--skip-done]
then: flock /tmp/gpuN.lock env CUDA_VISIBLE_DEVICES=N /home/hk28965/specwm/baselines/abot/run.sh --batch JOBS.json [--offload-t5 on]"""
import os, sys, json, glob, argparse
B = "/home/hk28965/specwm/baselines"
ap = argparse.ArgumentParser(); ap.add_argument("jobs"); ap.add_argument("--out-root", required=True); ap.add_argument("--set", default="abot", help="scene-set name used in the out-dir prefix")
ap.add_argument("--scenes-dir", default=None, help="default: %s/scenes/<set>" % B); ap.add_argument("--images", nargs="*", default=None, help="explicit image list instead of --scenes-dir")
ap.add_argument("--scenes", default="", help="[10/17] comma list of scene names to keep (the frozen subset, gen/frozen_scenes.py); default all images of the dir")
ap.add_argument("--programs", default="P1,P2,P3,P4,P5"); ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--no-caption", action="store_true", help="empty prompt for every job")
ap.add_argument("--skip-done", action="store_true", help="leave out jobs whose out dir already has video.mp4 + timing.json with frame_check_ok")
a = ap.parse_args()
imgs = a.images if a.images else sorted(glob.glob(os.path.join(a.scenes_dir or "%s/scenes/%s" % (B, a.set), "*.png")) + glob.glob(os.path.join(a.scenes_dir or "%s/scenes/%s" % (B, a.set), "*.jpg")))
assert imgs, "no images"
if a.scenes:
    keep = [s for s in a.scenes.split(",") if s]; byname = {os.path.splitext(os.path.basename(p))[0]: p for p in imgs}
    missing = [s for s in keep if s not in byname]; assert not missing, "--scenes not found in the scene dir: %s" % missing
    imgs = [byname[s] for s in sorted(dict.fromkeys(keep))]
jobs, skipped = [], 0
for img in imgs:
    name = os.path.splitext(os.path.basename(img))[0]; cap_f = os.path.splitext(img)[0] + ".txt"
    cap = "" if a.no_caption or not os.path.exists(cap_f) else open(cap_f).read().strip()
    for P in a.programs.split(","):
        out = os.path.join(a.out_root, "%s__%s__%s" % (a.set, name, P))
        if a.skip_done and os.path.exists(os.path.join(out, "video.mp4")) and os.path.exists(os.path.join(out, "timing.json")):
            try:
                if json.load(open(os.path.join(out, "timing.json"))).get("frame_check_ok"): skipped += 1; continue
            except Exception: pass
        jobs.append(dict(image=os.path.abspath(img), program=P, seed=a.seed, out=out, prompt=cap, scene=name, set=a.set))
os.makedirs(os.path.dirname(os.path.abspath(a.jobs)) or ".", exist_ok=True); json.dump(jobs, open(a.jobs, "w"), indent=1)
print("%s: %d jobs (%d images x %s, seed %d, captions %s)%s" % (a.jobs, len(jobs), len(imgs), a.programs, a.seed, "off" if a.no_caption else "from sibling .txt", "; %d already done skipped" % skipped if skipped else ""))
