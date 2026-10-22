"""f19: action-speculative generation on synthetic action streams (facts §28). Top row: per-frame action→frame latency CDF per stream family,
A arms (832 4-step: K = 0 / 1 / 2) from specwm/act_spec.py; bottom row: B arms (832 4-step under specwm/act_spec2.py: K0 fixed / K1 fixed /
K1 adaptive chunk length). Right column: output fps and generations per chunk. usage: envrun python paper/meeting_1012/scripts/f19_actspec.py"""
import glob, json, os, re, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = "/home/hk28965/specwm"; A = S + "/lingbot-world-v2/specwm_out/actspec"; B = S + "/lingbot-world-v2/specwm_out/actspec2"; OUT = S + "/paper/meeting_1012/figs/f19_actspec.png"
fam = lambda s: re.sub(r"_s\d+$", "", s); FAM = ["tau1", "tau2", "tau4", "tau2j03"]; TITLE = {"tau1": "τ = 1 s (change every 1 s)", "tau2": "τ = 2 s", "tau4": "τ = 4 s", "tau2j03": "τ = 2 s + 0.3° jitter"}
def load(root, arm):
    R = {}
    for p in glob.glob(root + "/*/*/K*_%s_s*/loop.json" % arm):
        if "/_warm/" in p: continue
        L = json.load(open(p)); R.setdefault((fam(L["stream"]), L["K"]), []).append(L)
    return R
def lat(Ls): return np.array([1000 * (t - L["chunk_start"][c]) for L in Ls for _, c, t in L["frame_ready"]])
rows = [("A: 832 4-step, fixed 3-frame chunks (act_spec)", load(A, "uniform4"), [(0, "K = 0 (no speculation)", "k"), (1, "K = 1", "C0"), (2, "K = 2", "C1")]),
        ("B: 832 4-step (act_spec2): fixed vs adaptive chunk length", {**{(f, "fx%d" % k): v for (f, k), v in load(B, "uniform4_fx").items()}, **{(f, "ad%d" % k): v for (f, k), v in load(B, "uniform4_ad3").items()}}, [("fx0", "K = 0 fixed", "k"), ("fx1", "K = 1 fixed", "C0"), ("ad1", "K = 1 adaptive (1-frame after a miss)", "C3")])]
fig, ax = plt.subplots(2, 5, figsize=(22, 8.5), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.1]))
for r, (title, R, arms) in enumerate(rows):
    for j, f in enumerate(FAM):
        a = ax[r, j]
        for key, lab, col in arms:
            Ls = R.get((f, key), [])
            if not Ls: continue
            v = np.sort(lat(Ls)); a.plot(v, np.arange(1, len(v) + 1) / len(v), color=col, label="%s (n = %d clips)" % (lab, len(Ls)))
        a.axvline(750, color="gray", ls=":", lw=1); a.set_xscale("log"); a.set_xlim(2, 8000); a.set_xlabel("action → frame latency (ms, log)"); a.set_title(TITLE[f] if r == 0 else ""); a.grid(alpha=0.3)
        if j == 0: a.set_ylabel("CDF over output frames"); a.legend(fontsize=8, loc="upper left")
    a = ax[r, 4]; xs = np.arange(len(FAM)); w = 0.8 / len(arms)
    for i, (key, lab, col) in enumerate(arms):
        fps = [np.mean([L["fps"] for L in R.get((f, key), [])]) if R.get((f, key)) else 0 for f in FAM]
        hr = [100 * np.mean([L["hit_rate"] for L in R.get((f, key), [])]) if R.get((f, key)) else 0 for f in FAM]
        a.bar(xs + (i - len(arms) / 2 + 0.5) * w, fps, w, color=col, label=lab)
        if key not in (0, "fx0"):
            for x, h, y in zip(xs, hr, fps): a.text(x + (i - len(arms) / 2 + 0.5) * w, y + 0.05, "%.0f%%" % h, ha="center", fontsize=7)
    a.set_xticks(xs); a.set_xticklabels(FAM); a.set_ylabel("output fps (sync decode on the same GPU)"); a.set_title("throughput (label = hit rate)"); a.grid(alpha=0.3, axis="y"); a.legend(fontsize=7)
    ax[r, 0].text(-0.12, 1.08, title, transform=ax[r, 0].transAxes, fontsize=11, fontweight="bold")
fig.suptitle("Action-speculative generation on synthetic key-press streams (6 held-out scenes; hit = pre-computed branch within 3° / relative translation 0.5 of the true action)", fontsize=12)
fig.tight_layout(); os.makedirs(os.path.dirname(OUT), exist_ok=True); fig.savefig(OUT, dpi=110); print("saved", OUT)
