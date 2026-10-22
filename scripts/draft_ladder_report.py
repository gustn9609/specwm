"""Draft (1.3B) speed ladder: specwm.fit24 --model 1p3b runs in lingbot-world-v2/specwm_out/draft_ladder/<steps>_<res>_<window>/ (FP8 weights, bf16 KV,
fused + partial commit, seed 1234, 43 bench-43 prompts). Prints: steady s/chunk, fps, chunks of look-ahead per real-time chunk (0.75 s / chunk_s),
peak memory, VLM quality / drift (draft_ladder_vlm.json) and the paired difference to the bench-43 student arm (uniform4_1p3b_s1234: 832, 4-step,
separate commit, bf16 KV) and to the teacher (uniform4_s1234). usage: python scripts/draft_ladder_report.py [--json OUT]"""
import sys, os, glob, json, numpy as np
S = "/home/hk28965/specwm"; O = S + "/lingbot-world-v2/specwm_out/draft_ladder"; B = S + "/lingbot-world-v2/specwm_out/bench43"
VL = {(k if k.startswith("/") else S + "/" + k): v for k, v in json.load(open(O + "/draft_ladder_vlm.json")).items()} if os.path.exists(O + "/draft_ladder_vlm.json") else {}
VB = {(k if k.startswith("/") else S + "/" + k): v for k, v in json.load(open(S + "/paper/1012/vlm_scores_bench43.json")).items()}
CF = [(st, res, w) for st in ("s4", "s2") for res in ("832", "704", "576") for w in ("w18", "w12")]
out = {}
print("| draft 1.3B config (FP8 w, bf16 KV, fused + partial commit) | steady s/chunk ↓ | fps ↑ | look-ahead per real-time chunk (0.75 s / s per chunk) ↑ | peak gen GiB ↓ | VLM quality ↑ | VLM drift ↓ | Δ quality vs student arm (paired, 43) ↑ | Δ drift vs student ↓ | Δ quality vs teacher ↑ |")
print("|---|---|---|---|---|---|---|---|---|---|")
for st, res, w in CF:
    k = "%s_%s_%s" % (st, res, w); p = O + "/" + k + "/fit.json"
    if not os.path.exists(p): print("| %s | – | – | – | – | – | – | – | – | – |" % k); continue
    r = json.load(open(p)); V = sorted([O + "/" + k + "/spec.mp4"] + glob.glob(O + "/" + k + "/*/spec.mp4"))
    q, d, dq, dd, dqt = [], [], [], [], []
    for v in V:
        t = os.path.basename(os.path.dirname(v)); t = "easy_lake" if t == k else t
        sp = "%s/%s/uniform4_1p3b_s1234/video.mp4" % (B, t); tp = "%s/%s/uniform4_s1234/video.mp4" % (B, t)
        if v in VL:
            q.append(VL[v]["quality"]); d.append(VL[v]["drift"])
            if sp in VB: dq.append(VL[v]["quality"] - VB[sp]["quality"]); dd.append(VL[v]["drift"] - VB[sp]["drift"])
            if tp in VB: dqt.append(VL[v]["quality"] - VB[tp]["quality"])
    se = lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0; f = lambda x, fmt: (fmt % (np.mean(x), se(x))) if x else "–"
    la = 0.75 / r["steady_chunk_s"]
    out[k] = dict(steps=4 if st == "s4" else 2, res=int(res), window=int(w[1:]), chunk_s=r["steady_chunk_s"], fps=r["fps"], lookahead=round(la, 2), peak_gen_GiB=r.get("peak_gen_GiB"), clips=len(V), q=float(np.mean(q)) if q else None, d=float(np.mean(d)) if d else None, dq=float(np.mean(dq)) if dq else None, dq_se=float(se(dq)) if dq else None, dd=float(np.mean(dd)) if dd else None, dqt=float(np.mean(dqt)) if dqt else None, dqt_se=float(se(dqt)) if dqt else None)
    print("| %d-step, %s, window %s | %.3f | %.1f | %.1f× | %.1f | %s | %s | %s | %s | %s |" % (out[k]["steps"], res, w[1:], r["steady_chunk_s"], r["fps"], la, r.get("peak_gen_GiB", float("nan")), f(q, "%.2f ± %.2f"), f(d, "%.1f ± %.1f"), f(dq, "%+.2f ± %.2f"), f(dd, "%+.1f ± %.1f"), f(dqt, "%+.2f ± %.2f")))
sp = [VB[k] for k in VB if "/uniform4_1p3b_s1234/" in k]; tp = [VB[k] for k in VB if "/uniform4_s1234/" in k]
print("\nreference arms (bench-43 seed 1234): student 1.3B 832 4-step separate commit: quality %.2f drift %.1f (n=%d); teacher 14B: %.2f / %.1f (n=%d)" % (np.mean([x["quality"] for x in sp]), np.mean([x["drift"] for x in sp]), len(sp), np.mean([x["quality"] for x in tp]), np.mean([x["drift"] for x in tp]), len(tp)))
if "--json" in sys.argv: json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
