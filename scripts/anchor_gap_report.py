"""Report for specwm/anchor_gap.py runs (lingbot-world-v2/specwm_out/anchor_gap/<task>_s<seed>/stats.json): per-chunk token gap to the 14B's
4-step output (sigma_T units, share <= eps) for the 14B's own steps 1-3 (T), a 14B sample from another noise (T2), the 1.3B anchored on the
teacher's history (A, 4 draft steps) and the 1.3B on its own history (O); AO = own-history draft vs anchored draft. Prints the chunk-index table
(KV-fill check), the per-latent-frame curves figure, and -- when anchor_gap_vlm.json exists -- the correlation of each clip's departure with the
VLM judge of the decoded student / teacher clip. usage: python scripts/anchor_gap_report.py [--json OUT] [--fig OUT.png]"""
import sys, os, glob, json, numpy as np
S = "/home/hk28965/specwm"; O = S + "/lingbot-world-v2/specwm_out/anchor_gap"; M = S + "/paper/meeting_1012"
runs = [json.load(open(p)) for p in sorted(glob.glob(O + "/*/stats.json"))]; NC = 13
runs = [r for r in runs if len(r["chunks"]) == NC]; print("runs: %d (tasks %d, seeds %s)" % (len(runs), len({r["task"] for r in runs}), sorted({r["seed"] for r in runs})))
def se(x): x = np.asarray(x, float); return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
def col(key, k=None, field="mean"):   # [runs, chunks]
    return np.array([[(ch[key][k] if k is not None else ch[key])[field] for ch in r["chunks"]] for r in runs])
def fcol(key, k=None, field="fmean"):   # [runs, frames]
    return np.array([np.concatenate([(ch[key][k] if k is not None else ch[key])[field] for ch in r["chunks"]]) for r in runs])
SER = [("14B step 1 (self-spec.)", "T", 0), ("14B step 2", "T", 1), ("14B step 3", "T", 2), ("14B, other noise", "T2", None), ("1.3B anchored, 4 steps", "A", 3), ("1.3B anchored, 1 step", "A", 0), ("1.3B own history, 4 steps", "O", None), ("own vs anchored draft", "AO", None)]
out = dict(n=len(runs), per_chunk={}, per_frame={})
print("\nPer chunk index (mean gap in sigma_T / share of tokens <= eps; mean over %d runs; chunk 0 = empty KV):" % len(runs))
print("| chunk | " + " | ".join(n for n, *_ in SER) + " |"); print("|---|" + "---|" * len(SER))
for c in range(NC):
    cells = []
    for n, key, k in SER:
        m = col(key, k, "mean")[:, c]; f = col(key, k, "frac")[:, c]; cells.append("%.3f / %.0f %%" % (m.mean(), 100 * f.mean())); out["per_chunk"].setdefault(n, []).append(dict(c=c, mean=float(m.mean()), mean_se=float(se(m)), frac=float(f.mean()), frac_se=float(se(f))))
    print("| %d | " % c + " | ".join(cells) + " |")
print("| mean c>=1 | " + " | ".join("%.3f / %.0f %%" % (col(key, k, "mean")[:, 1:].mean(), 100 * col(key, k, "frac")[:, 1:].mean()) for n, key, k in SER) + " |")
# how much of the own-history gap is history (vs capacity): O vs A at the same chunk
A4 = col("A", 3, "mean"); O4 = col("O", None, "mean"); AO = col("AO", None, "mean")
print("\nown-history gap minus anchored gap per chunk (sigma_T, mean ± se over runs):", " ".join("c%d %+.2f±%.2f" % (c, (O4 - A4)[:, c].mean(), se((O4 - A4)[:, c])) for c in range(NC)))
# per-frame figure
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for n, key, k in SER:
        if n in ("14B step 2", "14B step 3", "1.3B anchored, 1 step"): continue
        for j, fld in enumerate(("fmean", "ffrac")):
            y = fcol(key, k, fld); mu = y.mean(0); s = y.std(0, ddof=1) / np.sqrt(len(y)) if len(y) > 1 else 0 * mu
            ax[j].plot(np.arange(len(mu)), mu, label=n); ax[j].fill_between(np.arange(len(mu)), mu - s, mu + s, alpha=0.2)
            out["per_frame"].setdefault(n, {})[fld] = [float(v) for v in mu]
    for j, t in enumerate(("mean token gap to the 14B 4-step output (sigma_T)", "share of tokens within eps = 0.24")):
        ax[j].set_xlabel("latent frame (3 per chunk; chunk boundaries dotted)"); ax[j].set_title(t); ax[j].grid(alpha=0.3)
        for b in range(3, 3 * NC, 3): ax[j].axvline(b - 0.5, color="k", lw=0.4, ls=":", alpha=0.4)
    ax[0].legend(fontsize=8); fig.suptitle("Draft-to-teacher gap per latent frame, %d runs (43 bench-43 prompts x seeds), 832x480, 14B window 18" % len(runs), fontsize=10)
    fig.tight_layout(); fp = sys.argv[sys.argv.index("--fig") + 1] if "--fig" in sys.argv else M + "/figs/f15_anchor_gap.png"; fig.savefig(fp, dpi=150); print("\nfigure:", fp)
except Exception as e: print("figure skipped:", e)
# correlation with the VLM judge of the decoded clips
vp = O + "/anchor_gap_vlm.json"
if os.path.exists(vp):
    V = json.load(open(vp)); V = {(k if k.startswith("/") else S + "/" + k): v for k, v in V.items()}
    try: from scipy.stats import spearmanr
    except ImportError: spearmanr = None
    rows = []
    for r, a4, o4 in zip(runs, A4, O4):
        d = "%s/%s_s%d" % (O, r["task"], r["seed"]); sv, tv = V.get(d + "/student/spec.mp4"), V.get(d + "/teacher/spec.mp4")
        if sv and tv: rows.append(dict(task=r["task"], seed=r["seed"], q_s=sv["quality"], d_s=sv["drift"], q_t=tv["quality"], d_t=tv["drift"], o_all=o4[1:].mean(), o_early=o4[1:5].mean(), o_late=o4[9:].mean(), a_all=a4[1:].mean(), o_c0=o4[0], dq=sv["quality"] - tv["quality"], dd=sv["drift"] - tv["drift"]))
    if rows:
        print("\nVLM judge of the decoded clips (%d pairs): student %.2f / %.1f, teacher %.2f / %.1f (quality / drift)" % (len(rows), np.mean([x["q_s"] for x in rows]), np.mean([x["d_s"] for x in rows]), np.mean([x["q_t"] for x in rows]), np.mean([x["d_t"] for x in rows])))
        print("| departure measure (per clip) | Spearman vs student quality ↑ | vs student drift ↓ | vs Δquality (student − teacher) | vs Δdrift |"); print("|---|---|---|---|---|")
        for nm, key in (("own-history gap, chunks 1-12 mean", "o_all"), ("own-history gap, chunks 1-4", "o_early"), ("own-history gap, chunks 9-12", "o_late"), ("anchored gap, chunks 1-12 (capacity)", "a_all"), ("gap at chunk 0 (empty KV)", "o_c0")):
            x = [r_[key] for r_ in rows]; cells = []
            for tgt in ("q_s", "d_s", "dq", "dd"):
                y = [r_[tgt] for r_ in rows]
                if spearmanr: rho, p = spearmanr(x, y); cells.append("%+.2f (p=%.3f)" % (rho, p))
                else: cells.append("%+.2f" % np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])
            print("| %s | " % nm + " | ".join(cells) + " |")
        out["vlm_rows"] = rows
if "--json" in sys.argv: json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
