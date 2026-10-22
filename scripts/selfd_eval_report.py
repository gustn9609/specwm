"""Report for specwm/self_gap.py runs: node-0 self-distilled LoRA variants vs the stock 14B step-1 estimate on held-out scenes.
usage: python scripts/selfd_eval_report.py LABEL=DIR [LABEL=DIR ...] [--tasks a,b] [--seeds 7] [--json OUT]
Per variant (mean over chunks 1-12 and runs): stock step-1/2/3 within eps (reference, from the same runs), LoRA step-1 (L1) gap / within eps,
L1 vs stock step-1 (how far the LoRA moved it), LoRA-first-step 4-step output (L4) within eps of the stock final, and the VLM judge of the decoded
1-step-every-chunk rollouts (stock1, lora1; paired difference) with the teacher clip as control. Reads DIR/self_gap_vlm.json if present."""
import sys, os, json, glob, numpy as np
S = "/home/hk28965/specwm"
args = sys.argv[1:]; specs = [x for x in args if "=" in x and not x.startswith("--")]; opt = {args[i]: args[i + 1] for i in range(len(args) - 1) if args[i].startswith("--")}
tasks = set(opt["--tasks"].split(",")) if "--tasks" in opt else None; seeds = set(int(s) for s in opt.get("--seeds", "7").split(","))
def load(d):
    R = {}; vp = d + "/self_gap_vlm.json"; V = json.load(open(vp)) if os.path.exists(vp) else {}; V = {(k if k.startswith("/") else S + "/" + k): v for k, v in V.items()}
    for p in sorted(glob.glob(d + "/*/stats.json")):
        r = json.load(open(p))
        if (tasks and r["task"] not in tasks) or r["seed"] not in seeds or len(r["chunks"]) != 13: continue
        ch = r["chunks"][1:]; k = (r["task"], r["seed"]); dd = os.path.dirname(p); g = lambda nm: V.get(dd + "/" + nm + "/spec.mp4") or {}
        R[k] = dict(t1=np.mean([c["T"][0]["frac"] for c in ch]), t2=np.mean([c["T"][1]["frac"] for c in ch]), t3=np.mean([c["T"][2]["frac"] for c in ch]), t1m=np.mean([c["T"][0]["mean"] for c in ch]),
                    l1=np.mean([c["L1"]["frac"] for c in ch]), l1m=np.mean([c["L1"]["mean"] for c in ch]), l2=np.mean([c["L2"]["frac"] for c in ch]), l4=np.mean([c["L4"]["frac"] for c in ch]), l4m=np.mean([c["L4"]["mean"] for c in ch]), l1t1=np.mean([c["L1_vs_T1"]["mean"] for c in ch]),
                    q_s=g("stock1").get("quality", np.nan), d_s=g("stock1").get("drift", np.nan), q_l=g("lora1").get("quality", np.nan), d_l=g("lora1").get("drift", np.nan), q_t=g("teacher").get("quality", np.nan), d_t=g("teacher").get("drift", np.nan))
    return R
V = [(s.split("=", 1)[0], load(s.split("=", 1)[1])) for s in specs]
se = lambda x: np.nanstd(x, ddof=1) / np.sqrt(np.sum(~np.isnan(x))) if np.sum(~np.isnan(x)) > 1 else 0
print("| variant | n | stock step-1 within ε (ref) | stock step-2 | stock step-3 | LoRA step-1 gap σ_T ↓ | LoRA step-1 within ε ↑ | LoRA step-2 within ε ↑ | LoRA-first 4-step within ε of stock final ↑ | LoRA step-1 vs stock step-1 (σ_T) | 1-step clip quality: stock / LoRA ↑ | Δ quality (paired) ↑ | 1-step clip drift: stock / LoRA ↓ | Δ drift ↓ | teacher clip quality / drift (control) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
out = {}
for lab, R in V:
    ks = sorted(R); g = lambda f: np.array([R[k][f] for k in ks], float); dq = g("q_l") - g("q_s"); dd = g("d_l") - g("d_s")
    row = dict(n=len(ks), t1=g("t1").mean(), t2=g("t2").mean(), t3=g("t3").mean(), l1m=g("l1m").mean(), l1=g("l1").mean(), l2=g("l2").mean(), l4=g("l4").mean(), l1t1=g("l1t1").mean(), q_s=np.nanmean(g("q_s")), q_l=np.nanmean(g("q_l")), dq=np.nanmean(dq), dq_se=se(dq), d_s=np.nanmean(g("d_s")), d_l=np.nanmean(g("d_l")), dd=np.nanmean(dd), dd_se=se(dd), q_t=np.nanmean(g("q_t")), d_t=np.nanmean(g("d_t")))
    out[lab] = {k: float(v) for k, v in row.items()}
    print("| %s | %d | %.1f %% | %.1f %% | %.1f %% | %.3f | %.1f %% | %.1f %% | %.1f %% | %.3f | %.2f / %.2f | %+.2f ± %.2f | %.1f / %.1f | %+.1f ± %.1f | %.2f / %.1f |" % (lab, row["n"], 100 * row["t1"], 100 * row["t2"], 100 * row["t3"], row["l1m"], 100 * row["l1"], 100 * row["l2"], 100 * row["l4"], row["l1t1"], row["q_s"], row["q_l"], row["dq"], row["dq_se"], row["d_s"], row["d_l"], row["dd"], row["dd_se"], row["q_t"], row["d_t"]))
if "--json" in opt: json.dump(out, open(opt["--json"], "w"), indent=1)
