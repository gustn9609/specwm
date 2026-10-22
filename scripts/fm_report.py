"""[10/15] flow-map pilot report: VLM quality/drift per (label, N) from specwm_out/fm/vlm_fm.json + within-eps(jump after k, N-step path) from <evaldir>/eval[_base].json.
usage: python scripts/fm_report.py [evaldir1 evaldir2 ...]   (paths absolute; base13 = specwm_out/fm/eval_base13)"""
import json, os, sys, re, glob, numpy as np
O = "/home/hk28965/specwm/lingbot-world-v2/specwm_out/fm"; d = json.load(open(O + "/vlm_fm.json")); agg = {}
for p, v in d.items():
    m = re.search(r"/N(\d+)(_\w+)?/spec.mp4$", p)
    if not m or v.get("quality") is None: continue
    agg.setdefault((v["label"], int(m.group(1))), []).append((v["quality"], v["drift"]))
dirs = sys.argv[1:] or sorted(glob.glob(O + "/eval_*"))
wd = {}
for ed in dirs:
    for f in glob.glob(ed + "/eval*.json"):
        r = json.load(open(f)); lab = os.path.basename(ed).replace("eval_", "") + os.path.basename(f)[4:-5]
        for clip, byN in r.items():
            for N, w in byN.items(): wd.setdefault((lab, int(N[1:])), []).append([w.get("within_%d" % k, np.nan) for k in (1, 2, 3)])
labs = sorted({k[0] for k in agg} | {k[0] for k in wd})
print("| label | N | n | quality ↑ | drift ↓ | within-ε k=1 / 2 / 3 (jump-to-0 vs N-step path) ↑ |"); print("|---|---|---|---|---|---|")
for lab in labs:
    for N in sorted({k[1] for k in list(agg) + list(wd) if k[0] == lab}):
        q = agg.get((lab, N), []); w = wd.get((lab, N), [])
        qs = "%.2f | %.1f" % (np.mean([x[0] for x in q]), np.mean([x[1] for x in q])) if q else "– | –"
        ws = " / ".join("%.0f %%" % (100 * np.nanmean([x[i] for x in w])) if N > i + 1 else "–" for i in range(3)) if w else "–"
        print("| %s | %d | %d | %s | %s |" % (lab, N, len(q), qs, ws))
