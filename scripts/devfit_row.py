"""Final-table row for 'device-only 14B' = 704x384, FP8 weights, FP8 KV window, 2-step fused + partial commit, one GPU (fits 32 GB).
bench-43: seeds 1234 (fit24/q43_a704_fp8_fp8kv) + 4321 (fit24/q43s4321_a704_fp8_fp8kv), VLM scores in fit24/fit24_vlm.json, paired with the
teacher (uniform4) from paper/1012/vlm_scores_bench43.json; boundary jumps as bench43_report.py; gen s/clip from the clip loop.json
(root task: sum of chunk_s in fit.json); StEvo-38 physical inaccuracy from stevo38/eval_runs/lingbot_b38_dev704/summary.json.
usage: python scripts/devfit_row.py"""
import os, glob, json, numpy as np, sys
sys.path.insert(0, "/home/hk28965/specwm/scripts")
S = "/home/hk28965/specwm"; F = S + "/lingbot-world-v2/specwm_out/fit24"; B = S + "/lingbot-world-v2/specwm_out/bench43"
VL = {(k if k.startswith("/") else S + "/" + k): v for k, v in json.load(open(F + "/fit24_vlm.json")).items()}
VB = {(k if k.startswith("/") else S + "/" + k): v for k, v in json.load(open(S + "/paper/1012/vlm_scores_bench43.json")).items()}
def jumps(p):   # copy of bench43_report.jumps (importing it would run the whole report)
    import cv2
    cap = cv2.VideoCapture(p); prev = None; d = []
    while True:
        ok, f = cap.read()
        if not ok: break
        g = cv2.cvtColor(cv2.resize(f, (208, int(208 * f.shape[0] / f.shape[1]))), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None: d.append(np.abs(g - prev).mean())
        prev = g
    d = np.array(d); return np.array([d[b] / max(1e-6, np.mean([d[b - 2], d[b - 1], d[b + 1], d[b + 2]])) for b in range(20, len(d) - 2, 12)])
rows = []; J = []; gen = []
for seed, run in ((1234, "q43_a704_fp8_fp8kv"), (4321, "q43s4321_a704_fp8_fp8kv")):
    for p in sorted([F + "/" + run + "/spec.mp4"] + glob.glob(F + "/" + run + "/*/spec.mp4")):
        t = os.path.basename(os.path.dirname(p)); t = "easy_lake" if t == run else t
        tp = "%s/%s/uniform4_s%d/video.mp4" % (B, t, seed)
        if p in VL and tp in VB: rows.append((t, seed, VL[p]["quality"], VL[p]["drift"], VB[tp]["quality"], VB[tp]["drift"]))
        J.append(jumps(p))
        lj = os.path.dirname(p) + "/loop.json"
        if os.path.exists(lj): gen.append(json.load(open(lj))["gen_s"])
        elif t == "easy_lake": gen.append(sum(json.load(open(F + "/" + run + "/fit.json"))["chunk_s"]))
R = np.array([r[2:] for r in rows], float); se = lambda v: v.std(ddof=1) / np.sqrt(len(v))
q, d, tq, td = R.T; dq, dd = q - tq, d - td; J = np.concatenate(J)
fit = json.load(open(F + "/q43_a704_fp8_fp8kv/fit.json")); fps = fit["fps"]; st = F.replace("fit24", "stevo38") + "/eval_runs/lingbot_b38_dev704/summary.json"
phys = None
if os.path.exists(st):   # the eval writes per-task majority votes; 'overall' is filled by a later aggregation step that did not run here -> aggregate the tasks
    sj = json.load(open(st)); v = [list(t["llm_evals"].values())[0]["physical_inaccuracy"] for t in sj["tasks"]]; phys = sum(v) / len(v) if v else None
print("| arm | clips | gen s/clip (161 f) ↓ | steady fps (quiet) ↑ | VLM quality ↑ | VLM drift ↓ | Δ quality vs teacher (paired) ↑ | Δ drift vs teacher ↓ | boundary jumps > 1.5x ↓ | StEvo-38 phys. inaccuracy ↓ |")
print("|---|---|---|---|---|---|---|---|---|---|")
print("| device-only 14B: 704x384, FP8 weights, FP8 KV, 2-step fused + partial (fits 32 GB) | %d | %.1f | %.1f | %.2f ± %.2f | %.1f ± %.1f | %+.2f ± %.2f | %+.1f ± %.1f | %.0f %% | %s |" % (
    len(rows), np.median(gen), fps, q.mean(), se(q), d.mean(), se(d), dq.mean(), se(dq), dd.mean(), se(dd), 100 * (J > 1.5).mean(), ("%.2f" % phys) if phys is not None else "pending"))
print("quality better / equal / worse than the teacher: %d / %d / %d; teacher on the same pairs: quality %.2f drift %.1f" % ((dq > 0).sum(), (dq == 0).sum(), (dq < 0).sum(), tq.mean(), td.mean()))
json.dump(dict(clips=len(rows), gen_s=float(np.median(gen)), fps=fps, quality=float(q.mean()), quality_se=float(se(q)), drift=float(d.mean()), drift_se=float(se(d)), d_quality=float(dq.mean()), d_quality_se=float(se(dq)), d_drift=float(dd.mean()), d_drift_se=float(se(dd)), jump=float((J > 1.5).mean()), phys=phys), open(S + "/paper/meeting_1012/data/devfit_row.json", "w"), indent=1)
