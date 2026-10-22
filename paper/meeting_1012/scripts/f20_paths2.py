"""f20: rollout paths in latent space on the shared key-press stream tau2_s0 (facts §28h): per prompt, PCA of the generated frame latents over
all arms x 10 seeds; thin = seeds, thick = mean over seeds, dots = chunk boundaries (every 3 latent frames), X = last frame, circle = shared
first frame. Arms: target (14B 4-step), draft only (1.3B 4-step), 14B uniform 2-/3-step, ours teacher-parity (chunk-wise 0.7 + safe commit),
ours + action speculation (K = 1), ours + action speculation + adaptive 2-frame units. Latents are the generated latents (no decode / re-encode).
usage: envrun python paper/meeting_1012/scripts/f20_paths2.py"""
import glob, os, json, numpy as np, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = "/home/hk28965/specwm"; R = S + "/lingbot-world-v2/specwm_out/paths2"; OUT = S + "/paper/meeting_1012/figs/f20_paths2.png"; DATA = S + "/paper/meeting_1012/data/f20_paths2.json"
TASKS = ["sand_filling_up_jar", "sandcastle_wave_impact", "easy_lake", "easy_stonehenge_sunset", "pouring_water_into_cup"]; SEEDS = [1234, 4321, 9999, 11, 22, 33, 44, 55, 66, 77]
ARMS = [("target (14B, 4-step)", "k", lambda t, s: "%s/%s/tau2_s0/K0_uniform4P_fx_s%d/lat.pt" % (R, t, s)),
        ("draft only (1.3B, 4-step)", "tab:orange", lambda t, s: "%s/%s/uniform4_1p3b_tau2_s0_s%d/lat.pt" % (R, t, s)),
        ("14B uniform 2-step", "tab:blue", lambda t, s: "%s/%s/tau2_s0/K0_uniform2P_fx_s%d/lat.pt" % (R, t, s)),
        ("14B uniform 3-step", "tab:green", lambda t, s: "%s/%s/tau2_s0/K0_uniform3P_fx_s%d/lat.pt" % (R, t, s)),
        ("ours: teacher-parity (chunk-wise 0.7 + safe commit)", "tab:purple", lambda t, s: "%s/%s/tau2_s0/K0_cw0.7P_fx_s%d/lat.pt" % (R, t, s)),
        ("ours + action speculation (K = 1, 3-frame)", "tab:red", lambda t, s: "%s/%s/tau2_s0/K1_cw0.7P_fx_s%d/lat.pt" % (R, t, s)),
        ("ours + action speculation + adaptive 2-frame", "tab:brown", lambda t, s: "%s/%s/tau2_s0/K1_cw0.7P_ad3s2_s%d/lat.pt" % (R, t, s))]
fig, axes = plt.subplots(1, 5, figsize=(26, 5.6)); summary = {}
for ax, task in zip(axes, TASKS):
    X = []; idx = []   # all frame vectors of the prompt
    for ai, (nm, col, pf) in enumerate(ARMS):
        for s in SEEDS:
            p = pf(task, s)
            if not os.path.exists(p): continue
            L = torch.load(p).float(); F_ = L.shape[1]; v = L[:, :39].permute(1, 0, 2, 3).reshape(min(F_, 39), -1).numpy(); X.append(v); idx.append((ai, s, v.shape[0]))
    if not X: ax.set_title(task + " (no runs)"); continue
    Xa = np.concatenate(X, 0).astype(np.float32); mu = Xa.mean(0); Xc = torch.from_numpy(Xa - mu); U, Sv, V = torch.pca_lowrank(Xc, q=8, center=False); P = (Xc @ V[:, :2]).numpy(); ev = (Sv[:2] ** 2 / (Xc ** 2).sum()).tolist()
    o = 0; per_arm = {}
    for (ai, s, n), v in zip(idx, X): per_arm.setdefault(ai, []).append(P[o:o + n]); o += n
    nruns = {}
    for ai, (nm, col, pf) in enumerate(ARMS):
        if ai not in per_arm: continue
        paths = per_arm[ai]; nruns[nm] = len(paths)
        for pth in paths: ax.plot(pth[:, 0], pth[:, 1], color=col, lw=0.5, alpha=0.25)
        n = min(len(pth) for pth in paths); m = np.mean([pth[:n] for pth in paths], 0)
        ax.plot(m[:, 0], m[:, 1], color=col, lw=2.6, label="%s (n=%d)" % (nm, len(paths))); ax.scatter(m[2::3, 0], m[2::3, 1], color=col, s=10, zorder=3); ax.scatter(m[-1, 0], m[-1, 1], color=col, marker="x", s=70, lw=2, zorder=4)
    m0 = np.mean([pth[0] for paths in per_arm.values() for pth in paths], 0); ax.scatter(m0[0], m0[1], facecolors="none", edgecolors="k", s=90, lw=1.5, zorder=5); ax.annotate("start (shared first frame)", m0, textcoords="offset points", xytext=(8, -14), fontsize=8, color="0.3")
    ax.set_title("%s\nPC1 %.0f %%, PC2 %.0f %% of variance" % (task, 100 * ev[0], 100 * ev[1]), fontsize=10); ax.set_xlabel("PC 1 of frame latents"); ax.grid(alpha=0.3); summary[task] = dict(explained=ev, runs=nruns)
axes[0].set_ylabel("PC 2"); axes[0].legend(fontsize=7.5, loc="best")
fig.suptitle("Rollout paths in latent space on one key-press action stream (tau2_s0, change points at chunks 1/3/5/8/11/13); thin = seeds, thick = mean, dots = chunk boundaries, X = last frame", fontsize=11)
fig.tight_layout(); fig.savefig(OUT, dpi=110); json.dump(summary, open(DATA, "w"), indent=1); print("saved", OUT)
