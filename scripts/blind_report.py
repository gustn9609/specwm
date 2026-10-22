"""Summarize the blind pairwise study. First dump the answers: ArtifactData list collection 'ratings' with out_dir=paper/1012/blind/answers.
Each answer is stored relative to the candidate (+2 candidate clearly better than the 14B teacher … −2 teacher clearly better).
usage: python scripts/blind_report.py [paper/1012/blind/answers]"""
import sys, glob, json, collections, numpy as np
B = "/home/hk28965/specwm/paper/1012/blind"; d = sys.argv[1] if len(sys.argv) > 1 else B + "/answers"
trials = {t["id"]: t for t in json.load(open(B + "/trials.json"))}; mapping = json.load(open(B + "/mapping.json"))
R = collections.defaultdict(lambda: {"scene": [], "quality": []}); raters = 0
for f in glob.glob(d + "/**/*.json", recursive=True):
    doc = json.load(open(f)); ans = doc.get("answers") or doc.get("data", {}).get("answers") or {}
    if ans: raters += 1
    for tid, a in ans.items():
        t = trials.get(tid)
        if not t: continue
        arm = mapping[t["b"]]["arm"]; R[arm]["scene"].append(a["scene"]); R[arm]["quality"].append(a["quality"])
print("raters: %d" % raters)
print("| candidate (vs 14B 4-step teacher) | n | scene keeping: mean (−2…+2) ↑ | candidate ≥ teacher | visual quality: mean ↑ | candidate ≥ teacher |"); print("|---|---|---|---|---|---|")
for arm, v in sorted(R.items(), key=lambda kv: -np.mean(kv[1]["scene"])):
    s, q = np.array(v["scene"]), np.array(v["quality"])
    print("| %s | %d | %+.2f ± %.2f | %.0f %% | %+.2f ± %.2f | %.0f %% |" % (arm, len(s), s.mean(), s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0, 100 * (s >= 0).mean(), q.mean(), q.std(ddof=1) / np.sqrt(len(q)) if len(q) > 1 else 0, 100 * (q >= 0).mean()))
