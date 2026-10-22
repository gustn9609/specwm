"""Two-machine self-speculation (specwm/two_machine.py) report: timing per arm from the roles' timing.json, quality / drift from the VLM json,
paired against the solo 4-step teacher. usage: python scripts/twomach_report.py [TAG=a704] [OUTROOT]"""
import json, os, sys, glob, numpy as np
TAG = sys.argv[1] if len(sys.argv) > 1 else "a704"; O = (sys.argv[2] if len(sys.argv) > 2 else "/home/hk28965/specwm/lingbot-world-v2/specwm_out/twomach") + "/" + TAG
def med(v): v = sorted(v); return v[len(v) // 2] if v else float("nan")
def load(pattern): return {os.path.basename(os.path.dirname(p)): json.load(open(p)) for p in glob.glob(pattern)}
print("### timing (tag %s; 161 frames = 13 chunks of 0.75 s; medians over chunks 3.. of every clip)" % TAG)
print("| arm | role | clips | s per chunk ↓ (draft / total incl. re-commit) | fps shown ↑ | corrections received / applied per clip | correction lag ↓ (s, from show) | re-commit s | cloud refined / skipped per clip | cloud refine s | cloud-stream latency from show ↓ (s) | net one-way (s) | peak GiB |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for arm in sorted(d for d in os.listdir(O) if os.path.isdir(O + "/" + d)):
    dev = load(O + "/%s/device/*/timing.json" % arm); cl = load(O + "/%s/cloud/*/timing.json" % arm); solo = {k: v for k, v in (load(O + "/%s/*/timing.json" % arm) if not dev else {}).items() if "chunk_s" in v}
    if dev:
        ch = [r for t in dev.values() for r in t["chunks"][3:]]; co = [c for t in dev.values() for c in t["corrections"]]; ap_ = [c for c in co if c["applied"]]
        cc = [r for t in cl.values() for r in t["chunks"][3:]]; rf = [r for r in cc if r["mode"] == "refine"]
        print("| %s | device+cloud | %d | %.3f / %.3f | %.1f | %.1f / %.1f | %.2f | %.3f | %.1f / %.1f | %.3f | %.2f | %.4f | %.1f / %.1f |" % (
            arm, len(dev), med([r["draft_s"] for r in ch]), med([r["total_s"] for r in ch]), 12.0 / med([r["total_s"] for r in ch]), len(co) / len(dev), len(ap_) / len(dev), med([c["lag_s"] for c in ap_] or [float("nan")]), med([c["recommit_s"] for c in ap_] or [0]),
            len(rf) / max(1, len(cl)), (len(cc) - len(rf)) / max(1, len(cl)), med([r["work_s"] for r in rf] or [float("nan")]), med([r["lat_from_show_s"] for r in cc]), med([r["net_s"] for r in cc]), med([t["peak_GiB"] for t in dev.values()]), med([t["peak_GiB"] for t in cl.values()] or [0])))
    elif solo:
        ch = [s for t in solo.values() for s in t["chunk_s"][3:]]
        print("| %s | solo | %d | %.3f / – | %.1f | – | – | – | – | – | – | – | %.1f |" % (arm, len(solo), med(ch), 12.0 / med(ch), med([t["peak_GiB"] for t in solo.values()])))
V = {}
for p in glob.glob(os.path.dirname(O) + "/vlm_%s.json" % TAG): V = json.load(open(p))
if V:
    byl = {}
    for p, r in V.items(): byl.setdefault(r["label"], {})[p.split("/")[-2]] = r
    ref = byl.get(TAG + "_teacher", {})
    print("\n### quality (Qwen3-VL judge, 12 prompts seed 1234; paired Δ vs the solo FP8 4-step teacher; ↑ quality, ↓ drift)")
    print("| arm (label) | n | quality ↑ | drift ↓ | Δq vs teacher (paired) | Δdrift |"); print("|---|---|---|---|---|---|")
    for lab in sorted(byl):
        d = byl[lab]; ks = sorted(d); q = np.array([d[k]["quality"] for k in ks]); dr = np.array([d[k]["drift"] for k in ks]); c = [k for k in ks if k in ref]
        dq = np.array([d[k]["quality"] - ref[k]["quality"] for k in c]); dd = np.array([d[k]["drift"] - ref[k]["drift"] for k in c]); se = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
        print("| %s | %d | %.2f | %.1f | %+.2f ± %.2f | %+.1f ± %.1f |" % (lab, len(ks), q.mean(), dr.mean(), dq.mean() if len(c) else float("nan"), se(dq) if len(c) else float("nan"), dd.mean() if len(c) else float("nan"), se(dd) if len(c) else float("nan")))
