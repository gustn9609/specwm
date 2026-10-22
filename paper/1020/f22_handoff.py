"""f22 (clean): HANDOFF arms of the divergence probe (specwm/inject_handoff.py). Adapted from f21_clean.py.
Row 1: seed-MEAN rollout paths in the scene's latent PCA (fit on T + 1.3B-alone frames, all seeds): T black, 1.3B alone orange, hand@3m3 red,
       hand@3m1 pink, hand@6m2 purple, once@6 blue (reference). The 14B-generated segment of each handoff path is over-drawn in thick gray.
       Circle = shared start, X = last chunk.
Row 2: envelope ratio per chunk = |seed-mean path of the arm - seed-mean path of T| / mean seed-to-mean distance of T (2-D PCA, chunk = mean of its
       3 latent frames); < 1 = inside the 14B seed spread. Shaded bands = the 14B segments (chunks 4-6 for hand@3*, 7-8 for hand@6m2).
usage: envrun python /home/hk28965/specwm/paper/1020/f22_handoff.py"""
import os, sys, json, glob, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = "/home/hk28965/specwm"; R = S + "/lingbot-world-v2/specwm_out/inject/handoff"; RP = S + "/lingbot-world-v2/specwm_out/inject/1p3b_4step"; TD = S + "/lingbot-world-v2/specwm_out/paths2"
OUT = S + "/paper/1020/figs/f22_handoff_clean.png"
SCENES = ["sand_filling_up_jar", "sandcastle_wave_impact", "easy_lake", "easy_stonehenge_sunset", "pouring_water_into_cup"]
ARMS = [("alone", "1.3B alone", "tab:orange", RP, None), ("hand@3m3", "hand@3m3: 14B chunks 4-6 on the 1.3B context", "tab:red", R, (4, 6)), ("hand@3m1", "hand@3m1: 14B chunk 4", "hotpink", R, (4, 4)),
        ("hand@6m2", "hand@6m2: 14B chunks 7-8", "tab:purple", R, (7, 8)), ("hand@reject", "hand@reject: drift head -> 14B redoes flagged chunks", "tab:green", R, "RUN"), ("once@6", "once@6: 14B's own 0-6 injected (ref.)", "tab:blue", RP, (0, 6))]
MIN_SEEDS = 5
FPC = 3
def load(p): return torch.load(p, map_location="cpu").float().numpy()
def frames(x): return x.transpose(1, 0, 2, 3).reshape(x.shape[1], -1)
summary = {}
NS = [sc for sc in SCENES if len(glob.glob("%s/%s/hand@3m3_s*/lat.pt" % (R, sc))) >= MIN_SEEDS]
fig = plt.figure(figsize=(4.6 * len(NS) + 1, 8.6)); gs = fig.add_gridspec(2, len(NS), hspace=0.42, wspace=0.28); drawn = 0; H1 = {}; H2 = {}
for j, sc in enumerate(NS):
    seeds = sorted(int(os.path.basename(p).split("_s")[1]) for p in glob.glob("%s/%s/alone_s*" % (RP, sc)))
    if len(glob.glob("%s/%s/hand@3m3_s*/lat.pt" % (R, sc))) < MIN_SEEDS: print("skip", sc); continue
    T = {s: frames(load("%s/%s/tau2_s0/K0_uniform4P_fx_s%d/lat.pt" % (TD, sc, s))) for s in seeds}
    A = {a: {s: frames(load("%s/%s/%s_s%d/lat.pt" % (root, sc, a, s))) for s in seeds if os.path.exists("%s/%s/%s_s%d/lat.pt" % (root, sc, a, s))} for a, _, _, root, _ in ARMS}
    N = T[seeds[0]].shape[0]; C = N // FPC
    X = np.concatenate([T[s] for s in seeds] + [A["alone"][s] for s in A["alone"]], 0); mu = X.mean(0); Xc = X - mu
    U, Sv, Vt = np.linalg.svd(Xc, full_matrices=False); P = Vt[:2].T; ev = (Sv[:2] ** 2) / (Sv ** 2).sum(); proj = lambda f: (f - mu) @ P
    Tp = np.stack([proj(T[s]) for s in seeds]); Tm = Tp.mean(0); Tspread = np.linalg.norm(Tp - Tm[None], axis=2).mean(0); chunk = lambda v: v[:C * FPC].reshape(C, FPC, -1).mean(1)
    ax = fig.add_subplot(gs[0, j]); ax.plot(Tm[:, 0], Tm[:, 1], "-", color="k", lw=3, label="T: 14B 4-step alone", zorder=5)
    ax.scatter(Tm[0, 0], Tm[0, 1], s=90, facecolors="white", edgecolors="k", zorder=6); ax.plot(Tm[-1, 0], Tm[-1, 1], "x", color="k", ms=11, mew=3, zorder=6)
    ratios = {}; ns = {}
    for a, lab, col, root, seg in ARMS:
        if not A[a]: continue
        Ap = np.stack([proj(A[a][s]) for s in A[a]]); Am = Ap.mean(0); ns[a] = len(A[a])
        ratios[a] = (np.linalg.norm(chunk(Am) - chunk(Tm), axis=1) / chunk(Tspread[:, None])[:, 0]).tolist()
        ax.plot(Am[:, 0], Am[:, 1], "--" if a == "once@6" else "-", color=col, lw=2.4, label="%s (n=%d)" % (lab, len(A[a])), zorder=4); ax.plot(Am[-1, 0], Am[-1, 1], "x", color=col, ms=11, mew=3, zorder=6)
        if seg == "RUN":   # verifier-driven: gray diamonds at the chunks replaced by the 14B (size ~ share of seeds in which the chunk was replaced)
            rate = np.mean([json.load(open("%s/%s/%s_s%d/run.json" % (R, sc, a, s)))["replaced"] for s in A[a]], 0); summary.setdefault("replaced_rate", {})[sc] = rate.tolist()
            for c in range(C):
                if rate[c] > 0: ax.plot(Am[c * FPC + 1, 0], Am[c * FPC + 1, 1], "D", color="0.45", ms=4 + 8 * rate[c], mec="k", alpha=0.85, zorder=7)
            rr = ax2_rates = rate
        elif seg is not None and a.startswith("hand"):
            f0, f1 = seg[0] * FPC, (seg[1] + 1) * FPC; ax.plot(Am[f0 - 1:f1, 0], Am[f0 - 1:f1, 1], "-", color="0.45", lw=6, alpha=0.75, zorder=3, solid_capstyle="round"); ax.plot(Am[f0, 0], Am[f0, 1], "D", color=col, ms=8, mec="k", zorder=7)
    ax.set_title(sc.replace("_", " "), fontsize=13); ax.grid(alpha=0.3); ax.tick_params(labelsize=9)
    ax.set_xlabel("PC 1 (%.0f %%)" % (100 * ev[0]), fontsize=10); ax.set_ylabel("PC 2 (%.0f %%)" % (100 * ev[1]) if j == 0 else "", fontsize=10)
    H1.update({l: h for h, l in zip(*ax.get_legend_handles_labels())})
    ax2 = fig.add_subplot(gs[1, j]); cs = np.arange(C)
    for a, lab, col, root, seg in ARMS:
        if a in ratios: ax2.plot(cs, ratios[a], "--" if a == "once@6" else "-", color=col, lw=2.4, label=lab.split(":")[0])
    ax2.axvspan(3.5, 6.5, color="0.5", alpha=0.18, lw=0); ax2.axvspan(6.5, 8.5, color="tab:purple", alpha=0.10, lw=0)
    if "hand@reject" in ratios:
        rr = np.array(summary["replaced_rate"][sc])
        for c in range(C):
            if rr[c] > 0: ax2.plot(c, ratios["hand@reject"][c], "D", color="0.45", ms=4 + 8 * rr[c], mec="k", zorder=7)
    ax2.axhline(1.0, color="k", lw=1); ax2.set_yscale("log"); ax2.set_ylim(0.05, 20); ax2.set_xticks(range(0, C, 2)); ax2.grid(alpha=0.3, which="both"); ax2.tick_params(labelsize=9)
    ax2.set_xlabel("chunk (0.75 s each)", fontsize=10); ax2.set_ylabel("envelope ratio (< 1 = inside 14B seed spread)" if j == 0 else "", fontsize=10)
    H2.update({l: h for h, l in zip(*ax2.get_legend_handles_labels())})
    summary[sc] = dict(replaced_rate=summary.get("replaced_rate", {}).get(sc), explained=[float(e) for e in ev], ratios=ratios, n_seeds=ns); drawn += 1
fig.suptitle("Handoff: the 14B 4-step takes over for m chunks ON THE 1.3B's OWN CONTEXT, then the 1.3B continues (seed-mean paths; stream tau2_s0; thick gray = chunks generated by the 14B)", fontsize=13, y=0.995)
fig.axes[0].legend(H1.values(), H1.keys(), fontsize=7, loc="best", framealpha=0.9); fig.axes[len(NS)].legend(H2.values(), H2.keys(), fontsize=7, loc="lower left", framealpha=0.9)
fig.text(0.5, 0.005, "Row 1: seed-mean paths in the scene's latent PCA (fit on T + 1.3B-alone frames); circle = shared start, X = last chunk, thick gray = chunks generated by the 14B, diamond = first 14B chunk. Row 2: |arm seed-mean - T seed-mean| / mean T seed spread per chunk; gray band = 14B segment of hand@3m3 (hand@3m1: chunk 4 only), purple band = hand@6m2; gray diamonds = chunks the verifier handed to the 14B (size = share of seeds).", ha="center", fontsize=9)
fig.savefig(OUT, dpi=110, bbox_inches="tight"); json.dump(summary, open(OUT.replace(".png", ".json"), "w"), indent=1); print("wrote", OUT)
