"""[10/16] Report for the Speculative-Forcing-currency metrics (scripts/vbench6.py + scripts/rpe_da3.py JSONs): per-label means and PAIRED deltas vs a
reference label on the same (task, seed) clips. Keys: task = path component in tasks_camera.json (or <task>_s<seed>), seed = _s<seed> in the path else 1234.
usage: python scripts/metrics_report.py [--vb data/vbench6.json] [--rpe data/rpe_da3.json] [--ref teacher704] LABEL [LABEL ...]   (no labels = all)"""
import sys, os, re, json, argparse, numpy as np
S = "/home/hk28965/specwm"; P = S + "/paper/meeting_1012/data"
ap = argparse.ArgumentParser(); ap.add_argument("labels", nargs="*"); ap.add_argument("--vb", default=P + "/vbench6.json"); ap.add_argument("--rpe", default=P + "/rpe_da3.json"); ap.add_argument("--ref", default="teacher704"); ap.add_argument("--vlm", default="", help="optional vlm_clip_score JSON(s), comma-separated: adds judge quality/drift columns (labels must match)")
a = ap.parse_args()
TASKS = {t["id"] for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
def key(p):
    parts = p.split("/"); task = None
    for q in reversed(parts):
        if q in TASKS: task = q; break
        m = re.match(r"(.+?)_s\d+$", q)
        if m and m[1] in TASKS: task = m[1]; break
    m = re.search(r"_s(\d+)", p); return (task, int(m[1]) if m else 1234)
VB = json.load(open(a.vb)) if os.path.exists(a.vb) else {}; RP = json.load(open(a.rpe)) if os.path.exists(a.rpe) else {}
def by_label(D):
    out = {}
    for p, r in D.items(): out.setdefault(r["label"], {})[key(p)] = r
    return out
def ms(v): return "%.3f ± %.3f" % (np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else "%.3f" % np.mean(v)
VL = {}
for f in [x for x in a.vlm.split(",") if x]: VL.update({(k if k.startswith("/") else S + "/" + k): v for k, v in json.load(open(f)).items()})
vb, rp, vl = by_label(VB), by_label(RP), by_label(VL); labels = a.labels or sorted(set(vb) | set(rp) | set(vl))
if vl:
    print("| label | n | judge quality ↑ | judge drift ↓ | paired Δ quality vs %s (b/e/w) | Δ drift |" % a.ref); print("|---|---|---|---|---|---|")
    for l in labels:
        A = vl.get(l, {}); B = vl.get(a.ref, {}); ks = sorted(set(A) & set(B)); dq = np.array([A[k]["quality"] - B[k]["quality"] for k in ks]); dd = np.array([A[k]["drift"] - B[k]["drift"] for k in ks])
        if not A: continue
        print("| %s | %d | %.2f | %.1f | %s | %s |" % (l, len(A), np.mean([r["quality"] for r in A.values()]), np.mean([r["drift"] for r in A.values()]), ("%s (%d/%d/%d)" % (ms(dq), (dq > 0).sum(), (dq == 0).sum(), (dq < 0).sum())) if len(ks) and l != a.ref else "–", ms(dd) if len(ks) and l != a.ref else "–"))
    print()
F = [("quality_score", "VBench Q ↑", 2), ("imaging_quality", "Imaging ↑", 4), ("subject_consistency", "Subj ↑", 4), ("background_consistency", "Backg ↑", 4), ("motion_smoothness", "Motion ↑", 4), ("dynamic_degree", "Dyn ↑", 2), ("aesthetic_quality", "Aesth ↑", 4), ("musiq_drift", "Drift ↓", 3)]
def ms(v): return "%.3f ± %.3f" % (np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else "%.3f" % np.mean(v)
print("| label | n | " + " | ".join(f[1] for f in F) + " | n | RPE Rot ↓ | RPE Trans ↓ |"); print("|---|---|" + "---|" * (len(F) + 3))
for l in labels:
    rs = list(vb.get(l, {}).values()); rr = list(rp.get(l, {}).values())
    print("| %s | %d | " % (l, len(rs)) + " | ".join(("%%.%df" % f[2]) % np.mean([r[f[0]] for r in rs]) if rs else "–" for f in F) + " | %d | %s | %s |" % (len(rr), "%.3f" % np.mean([r["rot"] for r in rr]) if rr else "–", "%.3f" % np.mean([r["trans"] for r in rr]) if rr else "–"))
print("\nPaired vs %s (same task & seed; mean ± s.e.; better/equal(|Δ|<0.5 Q or <0.1°)/worse):" % a.ref)
print("| label | n | Δ VBench Q ↑ | Δ Imaging ↑ | Δ Drift ↓ | n | Δ Rot ↓ | Δ Trans ↓ |"); print("|---|---|---|---|---|---|---|---|")
for l in labels:
    if l == a.ref: continue
    A, B = vb.get(l, {}), vb.get(a.ref, {}); ks = sorted(set(A) & set(B)); RA, RB = rp.get(l, {}), rp.get(a.ref, {}); rk = sorted(set(RA) & set(RB))
    if not ks and not rk: continue
    dq = np.array([A[k]["quality_score"] - B[k]["quality_score"] for k in ks]); di = np.array([A[k]["imaging_quality"] - B[k]["imaging_quality"] for k in ks]); dd = np.array([A[k]["musiq_drift"] - B[k]["musiq_drift"] for k in ks])
    dr = np.array([RA[k]["rot"] - RB[k]["rot"] for k in rk]); dt = np.array([RA[k]["trans"] - RB[k]["trans"] for k in rk])
    bew = lambda d, eps, hi: "%d/%d/%d" % ((d > eps).sum() if hi else (d < -eps).sum(), (abs(d) <= eps).sum(), (d < -eps).sum() if hi else (d > eps).sum())
    print("| %s | %d | %s (%s) | %s | %s (%s) | %d | %s (%s) | %s |" % (l, len(ks), ms(dq) if len(ks) else "–", bew(dq, 0.5, True) if len(ks) else "", ms(di) if len(ks) else "–", ms(dd) if len(ks) else "–", bew(dd, 0.2, False) if len(ks) else "",
          len(rk), ms(dr) if len(rk) else "–", bew(dr, 0.1, False) if len(rk) else "", ms(dt) if len(rk) else "–"))
