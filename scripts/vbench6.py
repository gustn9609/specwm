"""[10/16] Speculative-Forcing-style quality metrics on our clips: VBench 6 quality dimensions (subject / background consistency, motion smoothness,
dynamic degree, aesthetic quality, imaging quality; temporal flickering excluded as in arXiv 2610.22347) + the VBench "Quality Score" aggregate
(official leaderboard normalization: min/max per dimension, weights 1 except dynamic degree 0.5; sanity: their Table 3 4-step row reproduces 82.6) +
MUSIQ Drift (Reward Forcing: std of the mean MUSIQ over non-overlapping 2 s = 32-frame segments, raw 0-100 scale).
Runs in tools/vbench_env (VBench pins transformers 4.33 / numpy < 2). Results are written per video (absolute path) with a label, like vlm_clip_score.
usage: tools/vbench_env/bin/python scripts/vbench6.py OUT.json LABEL=GLOB [LABEL=GLOB ...] [--device cuda:0] [--seg 32]"""
import sys, os, glob, json, argparse, shutil, re, numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("specs", nargs="+"); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--seg", type=int, default=32)
ap.add_argument("--tmp", default="/tmp/claude-1741758895/-home-hk28965/76e7a8ef-ae2e-455b-8c73-052f5bde2a30/scratchpad/vb_in"); ap.add_argument("--redo", action="store_true")
ap.add_argument("--fps-norm", type=float, default=0.0, help="[10/17 baselines, additive, default off] > 0: a clip whose container fps differs by > 1%% is resampled to this fps (ffmpeg fps filter, libx264 crf 10) before scoring, so the fps-sensitive dims (consistency, smoothness, dynamic degree) and the 2 s = 32-frame MUSIQ segments compare like with like across 60/20/16/12 fps models; results stay keyed by the original path (fields fps_src / fps_norm added)")
a = ap.parse_args()
import subprocess
def container_fps(v):
    try: n, d = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", v], capture_output=True, text=True, check=True).stdout.strip().split(",")[0].split("/"); return float(n) / float(d)
    except Exception: return 0.0
DIMS = ["subject_consistency", "background_consistency", "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality"]
NORM = {"subject_consistency": (0.1462, 1.0), "background_consistency": (0.2615, 1.0), "motion_smoothness": (0.706, 0.9975), "dynamic_degree": (0.0, 1.0),
        "aesthetic_quality": (0.0, 1.0), "imaging_quality": (0.0, 1.0)}   # VBench scripts/cal_final_score.py (temporal flickering (0.6293, 1.0) excluded)
W = {d: (0.5 if d == "dynamic_degree" else 1.0) for d in DIMS}
def quality_score(r):
    return 100 * sum(W[d] * (r[d] - NORM[d][0]) / (NORM[d][1] - NORM[d][0]) for d in DIMS) / sum(W.values())
OUT = json.load(open(a.out)) if os.path.exists(a.out) else {}
from vbench import VBench
from vbench.utils import load_video, CACHE_DIR, init_submodules
from vbench.imaging_quality import transform as iq_transform
from pyiqa.archs.musiq_arch import MUSIQ
dev = torch.device(a.device)
musiq = None
def musiq_frames(path):
    global musiq
    if musiq is None:
        sub = init_submodules(["imaging_quality"]); musiq = MUSIQ(pretrained_model_path=sub["imaging_quality"]["model_path"]).to(dev); musiq.training = False
    ims = iq_transform(load_video(path), "longer"); s = []
    with torch.no_grad():
        for i in range(len(ims)): s.append(float(musiq(ims[i].unsqueeze(0).to(dev))))
    return np.array(s)
for spec in a.specs:
    label, pat = spec.split("=", 1); vids = sorted(os.path.abspath(v) for v in glob.glob(pat)); assert vids, pat
    todo = [v for v in vids if a.redo or v not in OUT or OUT[v].get("label") != label or "quality_score" not in OUT[v]]
    print("[vbench6] %s: %d videos, %d to do" % (label, len(vids), len(todo)), flush=True)
    if not todo: continue
    d = os.path.join(a.tmp, label); shutil.rmtree(d, ignore_errors=True); os.makedirs(d); names = {}; used = {}
    for v in todo:
        n = re.sub(r"[^A-Za-z0-9]+", "_", os.path.relpath(v, "/home/hk28965/specwm")).strip("_") + ".mp4"; p = os.path.join(d, n); f = container_fps(v) if a.fps_norm > 0 else 0.0
        if a.fps_norm > 0 and f > 0 and abs(f - a.fps_norm) > 0.01 * a.fps_norm:   # resample to the common fps (keyed by the original path)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", v, "-vf", "fps=%g" % a.fps_norm, "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", p], check=True); used[v] = dict(path=p, fps_src=f, fps_norm=a.fps_norm)
        else: os.symlink(v, p); used[v] = dict(path=p)
        names[p] = v
    rd = os.path.join(a.tmp, "res_" + label); vb = VBench(dev, "", rd); vb.evaluate(videos_path=d, name=label, dimension_list=[x for x in DIMS if x != "imaging_quality"], mode="custom_input")
    res = json.load(open(os.path.join(rd, label + "_eval_results.json")))   # evaluate() returns None; {dim: [avg, [{video_path, video_results}]]}
    per = {v: {} for v in todo}
    for dim, (avg, vr) in res.items():
        for r in vr: per[names[r["video_path"]]][dim] = float(r["video_results"]) if dim != "dynamic_degree" else float(bool(r["video_results"]))
    for v in todo:
        s = musiq_frames(used[v]["path"]); n = len(s) // a.seg; segs = s[: n * a.seg].reshape(n, a.seg).mean(1)
        per[v]["imaging_quality"] = float(s.mean() / 100); per[v]["musiq_drift"] = float(segs.std()); per[v]["musiq_segments"] = [round(float(x), 2) for x in segs]
        if "fps_norm" in used[v]: per[v]["fps_src"] = used[v]["fps_src"]; per[v]["fps_norm"] = used[v]["fps_norm"]
        per[v]["quality_score"] = quality_score(per[v]); per[v]["label"] = label; OUT[v] = per[v]
        print("[vbench6] %s %s quality %.2f drift %.2f" % (label, os.path.relpath(v, "/home/hk28965/specwm"), per[v]["quality_score"], per[v]["musiq_drift"]), flush=True)
    json.dump(OUT, open(a.out, "w"), indent=1)
json.dump(OUT, open(a.out, "w"), indent=1)
# summary per label
for label in sorted({r["label"] for r in OUT.values()}):
    rs = [r for r in OUT.values() if r["label"] == label and "quality_score" in r]
    print("%-22s n=%3d quality %.2f  drift %.3f  " % (label, len(rs), np.mean([r["quality_score"] for r in rs]), np.mean([r["musiq_drift"] for r in rs])) + "  ".join("%s %.4f" % (d[:7], np.mean([r[d] for r in rs])) for d in DIMS))
