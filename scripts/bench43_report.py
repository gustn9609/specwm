"""Bench-43 report: 43 StEvo camera-prompt tasks x seeds 1234/4321, 8 arms. Reads the VLM scores (paper/1012/vlm_scores_bench43.json, written by
scripts/vlm_clip_score.py), the boundary-jump rate (scripts/boundary_jump.py logic, LingBot arms only: chunk = 12 frames) and timing from each
generator's own log. Prints English markdown tables: overall, paired vs the teacher, easy vs physics prompts, timing.
usage: python scripts/bench43_report.py [--json OUT.json]"""
import sys, glob, json, os, re, numpy as np, cv2
S = "/home/hk28965/specwm"; B = S + "/lingbot-world-v2/specwm_out/bench43"; P = S + "/paper/1012"
ARMS = [("teacher: 14B 4-step", "teacher", "uniform4_s*/video.mp4"), ("student: 1.3B 4-step", "student", "uniform4_1p3b_s*/video.mp4"),
        ("SDVG + actions, τ −0.7", "sdvg07", "ir_-0.7_s*/video.mp4"), ("SDVG + actions, τ −2.2", "sdvg22", "ir_-2.2_s*/video.mp4"),
        ("Matrix-Game 2.0", "mg2", "mg2_s*/video.mp4"), ("Waypoint-1.5-1B", "wp", "wp_s*/output.mp4"),
        ("ours: 832 fused + partial commit", "ours832", "pc1_832_s*/cloud/spec.mp4"), ("ours: 704 fused + partial commit", "ours704", "pc1_704_s*/cloud/spec.mp4"),
        ("chunk-wise 0.7, separate commit", "cw07_sep", "cw07_sep_s*/video.mp4"), ("chunk-wise 0.7, fused + partial", "cw07_pc1", "cw07_pc1_s*/video.mp4"),
        ("chunk-wise 0.7, safe commit", "cw07_safe", "cw07_safe_s*/video.mp4"), ("chunk-wise 0.5, fused + partial", "cw05_pc1", "cw05_pc1_s*/video.mp4")]
VLM = {(k if k.startswith("/") else S + "/" + k): v for k, v in json.load(open(P + "/vlm_scores_bench43.json")).items()} if os.path.exists(P + "/vlm_scores_bench43.json") else {}
def key(p):   # (task, seed)
    m = re.search(r"bench43/([^/]+)/[^/]*_s(\d+)/", p); return (m[1], int(m[2])) if m else None
def diffs(p):
    cap = cv2.VideoCapture(p); prev = None; d = []
    while True:
        ok, f = cap.read()
        if not ok: break
        g = cv2.cvtColor(cv2.resize(f, (208, int(208 * f.shape[0] / f.shape[1]))), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None: d.append(np.abs(g - prev).mean())
        prev = g
    return np.array(d)
def jumps(p):
    d = diffs(p); r = [d[b] / max(1e-6, np.mean([d[b - 2], d[b - 1], d[b + 1], d[b + 2]])) for b in range(20, len(d) - 2, 12)]; return np.array(r)
def chunk_s(arm, dirp):
    """steady seconds per 0.75 s chunk (one GPU), from the generator's own log"""
    try:
        if arm in ("teacher", "student") or arm.startswith("cw"):
            r = json.load(open(dirp + "/loop.json")); cs = np.diff(r["chunk_start"]); return float(np.median(cs[5:]))
        if arm in ("ours704", "ours832"):
            r = json.load(open(os.path.dirname(dirp) + "/loop.json")); return float(np.median(r["cloud_chunk_s"][6:]))
        if arm in ("sdvg07", "sdvg22"):
            r = json.load(open(dirp + "/loop.json")); return float(r["gen_s"]) / r["chunks"]
        if arm == "mg2":
            for f in glob.glob(dirp + "/*.json"):
                r = json.load(open(f))
                for k in ("fps", "gen_fps", "output_fps"):
                    if k in r: return 12.0 / float(r[k])
                if "gen_s" in r: return float(r["gen_s"]) / 13 * 153 / 156 * 13 / 13
        if arm == "wp":
            r = json.load(open(dirp + "/timing.json")); return 12.0 / float(r["output_fps"])
    except Exception: return None
rows = {}; per = {}
for lab, k, pat in ARMS:
    fs = sorted(glob.glob(B + "/*/" + pat)); sc = [(key(f), VLM[f]["drift"], VLM[f]["quality"]) for f in fs if f in VLM and key(f)]
    per[k] = {kk: (d, q) for kk, d, q in sc}
    J = np.concatenate([jumps(f) for f in fs]) if fs and k not in ("mg2", "wp") else np.array([])
    T = [chunk_s(k, os.path.dirname(f)) for f in fs]; T = [t for t in T if t]
    acc = []
    if k.startswith("sdvg"):
        for f in fs:
            try: acc.append(json.load(open(os.path.dirname(f) + "/loop.json"))["accepted"])
            except Exception: pass
    rows[k] = dict(label=lab, clips=len(fs), scored=len(sc), drift=[d for _, d, _ in sc], quality=[q for _, _, q in sc], jump=float((J > 1.5).mean()) if len(J) else None,
                   chunk_s=float(np.median(T)) if T else None, accepted=float(np.mean(acc)) if acc else None)
def ms(v): v = np.array(v, float); v = v[~np.isnan(v)]; return (v.mean(), v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else (float("nan"), 0)
print("| arm | clips | fps ↑ (one GPU, steady) | VLM drift ↓ | VLM quality ↑ | boundary jumps > 1.5× ↓ | SDVG chunks accepted |"); print("|---|---|---|---|---|---|---|")
out = {}
for lab, k, _ in ARMS:
    r = rows[k]; d = ms(r["drift"]); q = ms(r["quality"]); fps = 12 / r["chunk_s"] if r["chunk_s"] else float("nan")
    out[k] = dict(label=lab, clips=r["scored"], fps=round(fps, 1), drift=round(d[0], 1), drift_se=round(d[1], 1), quality=round(q[0], 2), quality_se=round(q[1], 2), jump=r["jump"], accepted=r["accepted"])
    print("| %s | %d | %.1f | %.1f ± %.1f | %.2f ± %.2f | %s | %s |" % (lab, r["scored"], fps, d[0], d[1], q[0], q[1], "%.0f %%" % (100 * r["jump"]) if r["jump"] is not None else "–", "%.1f / 13" % r["accepted"] if r["accepted"] is not None else "–"))
print("\nPaired with the teacher on the same task and seed (arm − teacher):"); print("| arm | pairs | Δ drift ↓ | Δ quality ↑ | quality: better / equal / worse |"); print("|---|---|---|---|---|")
for lab, k, _ in ARMS[1:]:
    ks = [x for x in per[k] if x in per["teacher"]]
    if len(ks) < 2: continue
    D = np.array([np.subtract(per[k][x], per["teacher"][x]) for x in ks]); dd, dq = ms(D[:, 0]), ms(D[:, 1])
    out[k].update(paired_n=len(ks), d_drift=round(dd[0], 1), d_drift_se=round(dd[1], 1), d_quality=round(dq[0], 2), d_quality_se=round(dq[1], 2))
    print("| %s | %d | %+.1f ± %.1f | %+.2f ± %.2f | %d / %d / %d |" % (lab, len(ks), dd[0], dd[1], dq[0], dq[1], (D[:, 1] > 0).sum(), (D[:, 1] == 0).sum(), (D[:, 1] < 0).sum()))
print("\nBy prompt group (quality / drift):"); print("| arm | easy scenes (5 prompts) | tabletop & physics (38 prompts) |"); print("|---|---|---|")
for lab, k, _ in ARMS:
    e = [v for x, v in per[k].items() if x[0].startswith("easy_")]; p_ = [v for x, v in per[k].items() if not x[0].startswith("easy_")]
    f = lambda L: "%.2f / %.0f (n=%d)" % (np.nanmean([q for _, q in L]), np.nanmean([d for d, _ in L]), len(L)) if L else "–"
    print("| %s | %s | %s |" % (lab, f(e), f(p_)))
if "--json" in sys.argv: json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
