"""Compare draft variants (stock 1.3B vs distilled LoRAs) on held-out scenes from anchor_gap.py runs.
usage: python scripts/distill_eval_report.py LABEL=DIR [LABEL=DIR ...] [--tasks a,b] [--seeds 7] [--vlm DIR/anchor_gap_vlm.json per DIR is read if present] [--json OUT]
Per variant: anchored gap (mean sigma_T / share <= eps) over chunks 1-12, own-history gap, and the VLM judge of the decoded own-history student
clip (quality / drift; blur check) with the paired difference to the first variant (= stock) on the same (task, seed)."""
import sys, os, json, glob, numpy as np
S = "/home/hk28965/specwm"
args = [x for x in sys.argv[1:]]; specs = [x for x in args if "=" in x and not x.startswith("--")]
opt = {args[i]: args[i + 1] for i in range(len(args) - 1) if args[i].startswith("--")}
tasks = set(opt["--tasks"].split(",")) if "--tasks" in opt else None; seeds = set(int(s) for s in opt.get("--seeds", "7").split(","))
def load(d):
    R = {}; vp = d + "/anchor_gap_vlm.json"; V = json.load(open(vp)) if os.path.exists(vp) else {}; V = {(k if k.startswith("/") else S + "/" + k): v for k, v in V.items()}
    for p in sorted(glob.glob(d + "/*/stats.json")):
        r = json.load(open(p))
        if (tasks and r["task"] not in tasks) or r["seed"] not in seeds or len(r["chunks"]) != 13: continue
        ch = r["chunks"][1:]; k = (r["task"], r["seed"]); sv = V.get(os.path.dirname(p) + "/student/spec.mp4"); tv = V.get(os.path.dirname(p) + "/teacher/spec.mp4")
        R[k] = dict(a_mean=np.mean([c["A"][3]["mean"] for c in ch]), a_frac=np.mean([c["A"][3]["frac"] for c in ch]), a1_frac=np.mean([c["A"][0]["frac"] for c in ch]), o_mean=np.mean([c["O"]["mean"] for c in ch]), o_frac=np.mean([c["O"]["frac"] for c in ch]),
                    t1_frac=np.mean([c["T"][0]["frac"] for c in ch]), t2_frac=np.mean([c["T2"]["frac"] for c in ch]), q_s=sv["quality"] if sv else np.nan, d_s=sv["drift"] if sv else np.nan, q_t=tv["quality"] if tv else np.nan, d_t=tv["drift"] if tv else np.nan)
    return R
V = [(s.split("=", 1)[0], load(s.split("=", 1)[1])) for s in specs]; base = V[0][1]
se = lambda x: np.nanstd(x, ddof=1) / np.sqrt(np.sum(~np.isnan(x))) if np.sum(~np.isnan(x)) > 1 else 0
print("| draft variant | n | anchored gap σ_T ↓ | anchored within ε ↑ | anchored 1-step within ε ↑ | own-history gap ↓ | own-history within ε ↑ | student clip VLM quality ↑ | student drift ↓ | Δ quality vs stock (paired) ↑ | Δ drift vs stock ↓ | teacher clip quality (control) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
out = {}
for lab, R in V:
    ks = sorted(R); g = lambda f: np.array([R[k][f] for k in ks], float); common = [k for k in ks if k in base]
    dq = np.array([R[k]["q_s"] - base[k]["q_s"] for k in common], float); dd = np.array([R[k]["d_s"] - base[k]["d_s"] for k in common], float)
    row = dict(n=len(ks), a_mean=g("a_mean").mean(), a_frac=g("a_frac").mean(), a1_frac=g("a1_frac").mean(), o_mean=g("o_mean").mean(), o_frac=g("o_frac").mean(), q_s=np.nanmean(g("q_s")), d_s=np.nanmean(g("d_s")), dq=np.nanmean(dq), dq_se=se(dq), dd=np.nanmean(dd), dd_se=se(dd), q_t=np.nanmean(g("q_t")))
    out[lab] = {k: float(v) for k, v in row.items()}
    print("| %s | %d | %.3f | %.1f %% | %.1f %% | %.3f | %.1f %% | %.2f | %.1f | %+.2f ± %.2f | %+.1f ± %.1f | %.2f |" % (lab, row["n"], row["a_mean"], 100 * row["a_frac"], 100 * row["a1_frac"], row["o_mean"], 100 * row["o_frac"], row["q_s"], row["d_s"], row["dq"], row["dq_se"], row["dd"], row["dd_se"], row["q_t"]))
r0 = V[0][1]; print("\nreference on the same clips: 14B own step 1 within ε %.1f %%, 14B other-noise sample %.1f %%" % (100 * np.mean([r0[k]["t1_frac"] for k in r0]), 100 * np.mean([r0[k]["t2_frac"] for k in r0])))
if "--json" in opt: json.dump(out, open(opt["--json"], "w"), indent=1)
