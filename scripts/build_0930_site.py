"""0930 meeting site: every clip in paper/0930_ppt (individual clips, not the baked side-by-sides) transcoded small + one HTML with synced players.
usage: build_0930_site.py OUT   (writes OUT/v/*.mp4 and OUT/meeting_0930.html)"""
import os, sys, json, subprocess, re, html
from concurrent.futures import ThreadPoolExecutor
S = "/home/hk28965/specwm"; P = S + "/paper/0930_ppt"; OUT = sys.argv[1]; os.makedirs(OUT + "/v", exist_ok=True)
FOLDERS = ["ar_vs_bidirectional", "sdvg_original_3methods", "sdvg_original_30s", "all_v2_pipeline"]
def parse(folder, fn):
    parts = fn[:-4].split("__")
    if any(p.startswith("side-by-side") for p in parts): return None
    task = parts[0]; seed = next((int(p[4:]) for p in parts if re.match(r"seed\d+$", p)), None)
    si = next(i for i, p in enumerate(parts) if p.startswith("seed")); lab = parts[si + 1]
    idx, label = lab.split("_", 1) if "_" in lab else (lab, lab)
    sec = next((int(p[:-1]) for p in parts if re.match(r"\d+s$", p)), None)
    frames = next((int(p[:-6]) for p in parts if p.endswith("frames")), None)
    return dict(folder=folder, task=task, seed=seed, idx=idx, label=label.replace("_", " "), sec=sec, frames=frames, src=os.path.join(P, folder, fn), fn="v/%s_%s_%d_%s.mp4" % (folder[:6], task, seed, idx))
recs = [r for f in FOLDERS for r in (parse(f, fn) for fn in sorted(os.listdir(os.path.join(P, f))) if fn.endswith(".mp4")) if r]
def transcode(r):
    dst = os.path.join(OUT, r["fn"])
    if not (os.path.exists(dst) and os.path.getmtime(dst) > os.path.getmtime(r["src"])):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", r["src"], "-vf", "scale=448:-2", "-r", "16", "-c:v", "libx264", "-preset", "medium", "-crf", "30", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst], check=True)
    r["bytes"] = os.path.getsize(dst); return r
with ThreadPoolExecutor(8) as ex: recs = list(ex.map(transcode, recs))
ACC = {}
for f in ("sdvg_original_3methods", "sdvg_original_30s"):
    for ln in open(os.path.join(P, f, "README.md")):
        m = re.match(r"\| (\w+)(?: s(\d+))? \| (\S+) \| [\d.]+ s \| (\d+) / (\d+) \|", ln)
        if m: ACC[(f, m.group(1), int(m.group(2) or 1234), m.group(3).split("_", 1)[0])] = "%s accepted / %s rejected" % (m.group(4), m.group(5))
for r in recs:
    del r["src"]; a = ACC.get((r["folder"], r["task"], r["seed"], r["idx"]))
    if a: r["accept"] = a
DATA = dict(clips=recs)
tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "site_0930_template.html")).read()
open(os.path.join(OUT, "meeting_0930.html"), "w").write(tpl.replace("/*__DATA__*/null", json.dumps(DATA)))
print("clips %d, %.1f MB" % (len(recs), sum(r["bytes"] for r in recs) / 1e6))
json.dump(recs, open(os.path.join(OUT, "clips.json"), "w"), indent=0)
