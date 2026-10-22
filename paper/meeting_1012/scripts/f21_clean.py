"""f21 (clean, meeting version): divergence injection, 1.3B <- 14B 4-step latents (specwm/inject_probe.py, facts §52).
Row 1: seed-MEAN rollout paths in the scene's latent PCA (fit on T + G-alone frames, 10 seeds): T (14B 4-step), G alone (1.3B 4-step),
       inject once at chunk 6, inject every chunk. Diamond = injection chunk, circle = shared start, X = last chunk. No per-seed lines.
Row 2: envelope ratio per chunk = |seed-mean path of the arm - seed-mean path of T| / mean seed-to-mean distance of T (2-D PCA, averaged over the
       chunk's 3 latent frames); < 1 = inside the 14B seed spread. Vertical dotted lines = injection chunks 3 / 6 / 9.
Side panel: follow ratio (facts §52 definition, mean over scenes +- s.e. over scenes) per arm.
usage: envrun python paper/meeting_1012/scripts/f21_clean.py [GEN=1p3b_4step|14b_2step]"""
import os, sys, json, glob, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = "/home/hk28965/specwm"; GEN = sys.argv[1] if len(sys.argv) > 1 else "1p3b_4step"
R = S + "/lingbot-world-v2/specwm_out/inject/" + GEN; TD = S + "/lingbot-world-v2/specwm_out/paths2"
OUT = S + "/paper/meeting_1012/figs/f21_inject_clean_%s.png" % GEN
SCENES = ["sand_filling_up_jar", "sandcastle_wave_impact", "easy_lake", "easy_stonehenge_sunset", "pouring_water_into_cup"]
GLABEL = {"1p3b_4step": "1.3B 4-step", "14b_2step": "14B 2-step"}[GEN]
ARMS = [("alone", "G alone (%s)" % GLABEL, "tab:orange", "-"), ("once@3", "inject once @ chunk 3", "tab:purple", "-"), ("once@6", "inject once @ chunk 6", "tab:blue", "-"),
        ("once@9", "inject once @ chunk 9", "tab:brown", "-"), ("every", "inject every chunk (ceiling)", "tab:green", "--")]
FPC = 3
def load(p): return torch.load(p, map_location="cpu").float().numpy()   # (16, 39, h, w)
def frames(x): return x.transpose(1, 0, 2, 3).reshape(x.shape[1], -1)   # (39, D)
summary = {}; follow = {}
fig = plt.figure(figsize=(23, 8.6)); gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 1, 0.9], hspace=0.42, wspace=0.28)
for j, sc in enumerate(SCENES):
    seeds = sorted(int(os.path.basename(p).split("_s")[1]) for p in glob.glob("%s/%s/alone_s*" % (R, sc)))
    T = {s: frames(load("%s/%s/tau2_s0/K0_uniform4P_fx_s%d/lat.pt" % (TD, sc, s))) for s in seeds}
    A = {a: {s: frames(load("%s/%s/%s_s%d/lat.pt" % (R, sc, a, s))) for s in seeds if os.path.exists("%s/%s/%s_s%d/lat.pt" % (R, sc, a, s))} for a, *_ in ARMS}
    N = T[seeds[0]].shape[0]; C = N // FPC
    X = np.concatenate([T[s] for s in seeds] + [A["alone"][s] for s in A["alone"]], 0); mu = X.mean(0); Xc = X - mu
    U, Sv, Vt = np.linalg.svd(Xc, full_matrices=False); P = Vt[:2].T; ev = (Sv[:2] ** 2) / (Sv ** 2).sum()
    proj = lambda f: (f - mu) @ P
    Tp = np.stack([proj(T[s]) for s in seeds]); Tm = Tp.mean(0); Tspread = np.linalg.norm(Tp - Tm[None], axis=2).mean(0)   # (N,), (N,2), (N,)
    chunk = lambda v: v[:C * FPC].reshape(C, FPC, -1).mean(1)
    ax = fig.add_subplot(gs[0, j]); ax.plot(Tm[:, 0], Tm[:, 1], "-", color="k", lw=3, label="T: 14B 4-step", zorder=5)
    ax.scatter(Tm[0, 0], Tm[0, 1], s=90, facecolors="white", edgecolors="k", zorder=6); ax.plot(Tm[-1, 0], Tm[-1, 1], "x", color="k", ms=11, mew=3, zorder=6)
    ratios = {}
    for a, lab, col, ls in ARMS:
        if not A[a]: continue
        Ap = np.stack([proj(A[a][s]) for s in A[a]]); Am = Ap.mean(0)
        ratios[a] = (np.linalg.norm(chunk(Am) - chunk(Tm), axis=1) / chunk(Tspread[:, None])[:, 0]).tolist()
        if a in ("alone", "once@6", "every"):
            ax.plot(Am[:, 0], Am[:, 1], ls, color=col, lw=2.6, label=lab, zorder=4); ax.plot(Am[-1, 0], Am[-1, 1], "x", color=col, ms=11, mew=3, zorder=6)
            if a == "once@6": k = 6; ax.plot(Am[k * FPC, 0], Am[k * FPC, 1], "D", color=col, ms=10, mec="k", zorder=7)
    ax.set_title(sc.replace("_", " "), fontsize=13); ax.grid(alpha=0.3); ax.tick_params(labelsize=9)
    ax.set_xlabel("PC 1 (%.0f %%)" % (100 * ev[0]), fontsize=10); ax.set_ylabel("PC 2 (%.0f %%)" % (100 * ev[1]) if j == 0 else "", fontsize=10)
    if j == 0: ax.legend(fontsize=9, loc="best", framealpha=0.9)
    ax2 = fig.add_subplot(gs[1, j]); cs = np.arange(C)
    for a, lab, col, ls in ARMS:
        if a in ratios: ax2.plot(cs, ratios[a], ls, color=col, lw=2.4, label=lab)
    for k in (3, 6, 9): ax2.axvline(k, color="gray", ls=":", lw=1.2)
    ax2.axhline(1.0, color="k", lw=1); ax2.set_yscale("log"); ax2.set_ylim(0.05, 20); ax2.set_xticks(range(0, C, 2)); ax2.grid(alpha=0.3, which="both"); ax2.tick_params(labelsize=9)
    ax2.set_xlabel("chunk (0.75 s each)", fontsize=10); ax2.set_ylabel("envelope ratio (< 1 = inside 14B seed spread)" if j == 0 else "", fontsize=10)
    if j == 0: ax2.legend(fontsize=9, loc="upper right", framealpha=0.9)
    summary[sc] = dict(explained=[float(e) for e in ev], ratios=ratios, n_seeds=len(seeds))
# side panel: follow ratio per arm from the probe's json (mean +- s.e. over scenes)
J = json.load(open(S + "/paper/meeting_1012/data/f21_inject.json")).get(GEN, {})
names = ["once@1", "once@3", "once@6", "once@9", "every", "self@3"]; vals = {n: [] for n in names}
for sc in SCENES:
    d = J.get(sc, {}).get("arms", {})
    for n in names:
        if n in d and "follow" in d[n]: vals[n].append(d[n]["follow"])
        elif n in d and "dT_mean" in d[n]:   # recompute: mean over chunks after k of dG/(dT+dG)
            k = d[n]["k"]; dT = np.array(d[n]["dT_mean"], float); dG = np.array(d[n]["dG_mean"], float); m = np.arange(len(dT)) > k
            m &= np.isfinite(dT) & np.isfinite(dG); vals[n].append(float((dG[m] / (dT[m] + dG[m])).mean()))
axb = fig.add_subplot(gs[:, 5]); y = np.arange(len(names))[::-1]
means = [np.mean(vals[n]) if vals[n] else np.nan for n in names]; ses = [np.std(vals[n], ddof=1) / np.sqrt(len(vals[n])) if len(vals[n]) > 1 else 0 for n in names]
cols = ["tab:red", "tab:purple", "tab:blue", "tab:brown", "tab:green", "0.6"]
labels = ["inject once @1", "inject once @3", "inject once @6", "inject once @9", "inject every chunk", "re-commit own latents @3"]
axb.barh(y, means, xerr=ses, color=cols, edgecolor="k", capsize=4, height=0.62); axb.set_yticks([]); axb.set_ylim(-0.7, len(names) - 0.3)
axb.set_xlim(0, 1); axb.axvline(0.5, color="k", ls=":", lw=1); axb.set_xlabel("follow ratio\n1 = on the 14B same-seed path, 0 = on its own path", fontsize=9); axb.set_title("follow ratio, mean over 5 scenes", fontsize=11); axb.grid(alpha=0.3, axis="x")
for yi, m, lab in zip(y, means, labels): axb.text(0.02, yi + 0.46, lab, va="center", ha="left", fontsize=10); axb.text(min(m + 0.02, 0.86), yi, "%.2f" % m, va="center", fontsize=10, fontweight="bold")
fig.suptitle("Divergence injection: %s continues after its context is replaced by the 14B 4-step latents (same seed, same key-press stream tau2_s0; 10 seeds per scene)" % GLABEL, fontsize=14, y=0.995)
fig.savefig(OUT, dpi=110, bbox_inches="tight"); json.dump(dict(summary=summary, follow=vals), open(OUT.replace(".png", ".json"), "w"), indent=1); print("wrote", OUT)
