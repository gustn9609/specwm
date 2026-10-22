"""10-seed version of the rollout-path figure (paper/1012/fig_paths10_<scene>.png).
Left : per-frame VAE latents of every arm and seed projected with ONE shared PCA; thin lines = single seeds, thick line = the per-frame mean over
       seeds (start = circle, X = mean last frame).
Right: relative latent distance of each arm's frame to the 14B 4-step frame of the SAME seed, mean over seeds with a ±1 s.e. band; the grey band
       is the 14B 4-step vs 14B 4-step with a DIFFERENT seed (how far two runs of the reference itself drift apart) -- anything inside it is
       within the reference's own run-to-run variability.
usage: python scripts/viz_paths10.py [--scene sand_filling_up_jar] [--seeds 1234,4321,9999,11,22,33,44,55,66,77]"""
import argparse, os, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
ap = argparse.ArgumentParser(); ap.add_argument("--scene", default="sand_filling_up_jar"); ap.add_argument("--seeds", default="1234,4321,9999,11,22,33,44,55,66,77"); ap.add_argument("--out", default="/home/hk28965/specwm/paper/1012"); a = ap.parse_args()
S = "/home/hk28965/specwm"; O = S + "/lingbot-world-v2/specwm_out"; SEEDS = [int(s) for s in a.seeds.split(",")]
C = dict(target="#1f1f1f", draft="#e07b00", uniform2="#2a7fd4", uniform3="#2aa198", early="#b03aa6", chunk="#7b4bd1", anchor="#c2185b", lowres="#6d8b00")
ARMS = [("target (14B, 4-step)", "target", O + "/early_cmp/%s/uniform4_s%d/lat.pt"), ("draft only (1.3B, 4-step)", "draft", O + "/pipe_ff3/%s/draft_1p3b_s%d/lat.pt"),
        ("14B uniform 2-step", "uniform2", O + "/early_cmp/%s/uniform2_s%d/lat.pt"), ("14B uniform 3-step", "uniform3", O + "/early_cmp/%s/uniform3_s%d/lat.pt"),
        ("ours: token early exit (2.3 steps)", "early", O + "/early_cmp/%s/early0.24_flash_s%d/lat.pt"), ("ours: chunk-wise adaptive (2.9 steps)", "chunk", O + "/early_cmp/%s/diag_learned_dense_0.24_chunk0.7_s%d/lat.pt"),
        ("ours: anchor split (14B step 1 + 1.3B 2 steps, 2 GPU)", "anchor", O + "/anchor_pipe/%s/anchor_d2_acceptall_s%d/lat.pt"), ("ours: 14B 2-step at 576×320 + SR (realtime)", "lowres", O + "/commit_pipe_lr576_sr/%s/local_u2_area175104_s%d/lat.pt")]
L = {}   # key -> {seed: [F, D] float32}
for lab, key, pat in ARMS:
    for s in SEEDS:
        p = pat % (a.scene, s)
        if os.path.exists(p): x = torch.load(p).float(); L.setdefault(key, {})[s] = x
HW = next(iter(L["target"].values())).shape[2:]
for d in L.values():   # re-encoded (SR) latents can differ by a few latent pixels: resample to the target grid
    for s, x in d.items():
        if tuple(x.shape[2:]) != tuple(HW): x = torch.nn.functional.interpolate(x.permute(1, 0, 2, 3), size=tuple(HW), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
        d[s] = x.permute(1, 0, 2, 3).reshape(x.shape[1], -1).numpy()
LAB = {k: lab for lab, k, _ in ARMS}; F = min(v.shape[0] for d in L.values() for v in d.values())
for d in L.values():
    for s in d: d[s] = d[s][:F]
print({k: sorted(d) for k, d in L.items()})
pca = PCA(2, random_state=0).fit(np.concatenate([v for d in L.values() for v in d.values()])); Z = {k: {s: pca.transform(v) for s, v in d.items()} for k, d in L.items()}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw=dict(width_ratios=[1.05, 1])); fig.patch.set_facecolor("white")
for ax in (ax1, ax2): ax.set_facecolor("white"); [ax.spines[s].set_visible(False) for s in ("top", "right")]; ax.grid(True, color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
order = [k for _, k, _ in ARMS if k in Z]
for k in order:
    col = C[k]; n = len(Z[k]); big = k in ("early", "chunk", "anchor", "lowres", "target")
    for s, z in Z[k].items(): ax1.plot(z[:, 0], z[:, 1], color=col, lw=0.5, alpha=0.10, zorder=2)
    m = np.mean(np.stack(list(Z[k].values())), 0); ax1.plot(m[:, 0], m[:, 1], color=col, lw=2.8 if big else 2.0, zorder=4 if big else 3, label="%s  (n=%d)" % (LAB[k], n))
    ax1.scatter(m[::3, 0], m[::3, 1], s=12, color=col, zorder=5, lw=0); ax1.scatter(m[-1, 0], m[-1, 1], s=80, color=col, marker="X", zorder=6, lw=0)
m0 = np.mean(np.stack(list(Z[order[0]].values())), 0)[0]; ax1.scatter(*m0, s=120, facecolor="white", edgecolor="#1f1f1f", lw=1.8, zorder=7); ax1.annotate("start (shared first frame)", m0, xytext=(8, 8), textcoords="offset points", fontsize=9, color="#444")
ax1.set_xlabel("PC 1 of frame latents (shared PCA over all arms and seeds)"); ax1.set_ylabel("PC 2"); ax1.legend(frameon=False, fontsize=8.5, loc="best")
ax1.set_title("Rollout paths in latent space — %s\n(thin = single seeds, thick = mean over seeds, dots = chunk boundaries, X = last frame)" % a.scene.replace("_", " "), fontsize=10.5, loc="left")
# right: distance to the same-seed target frame; grey band = target vs target with a different seed
def rel(x, t): return np.linalg.norm(x - t, axis=1) / np.linalg.norm(t, axis=1)
T = L["target"]; x = np.arange(F) / 3.0   # latent frames: 3 per chunk (= 12 video frames)
def band(ax, D, col, lab, lw, ls="-", z=3, fill=True):
    D = np.stack(D); m = D.mean(0); se = D.std(0, ddof=1) / np.sqrt(len(D)) if len(D) > 1 else 0 * m
    ax.plot(x, m, color=col, lw=lw, ls=ls, label="%s  (n=%d)" % (lab, len(D)), zorder=z)
    if fill: ax.fill_between(x, m - se, m + se, color=col, alpha=0.15, lw=0, zorder=z - 1)
    return m
tt = [rel(T[s2], T[s1]) for s1 in T for s2 in T if s1 != s2]
band(ax2, tt, "#8a8a8a", "14B 4-step vs 14B 4-step, different seed (reference's own spread)", 1.6, "--", 2)
summ = {}
for k in order:
    if k == "target": continue
    D = [rel(L[k][s], T[s]) for s in L[k] if s in T]
    if D: summ[k] = band(ax2, D, C[k], LAB[k], 2.6 if k in ("early", "chunk", "anchor", "lowres") else 1.7, z=4 if k in ("early", "chunk", "anchor", "lowres") else 3).mean()
for c in range(1, F // 3 + 1): ax2.axvline(c, color="#eeeeee", lw=0.8, zorder=1)
ax2.set_xlabel("chunk index (3 latent frames = 12 video frames = 0.75 s)"); ax2.set_ylabel("relative latent distance to the same-seed 14B 4-step frame ↓"); ax2.set_ylim(bottom=0)
ax2.set_title("Distance to the 4-step target frame over time (mean ± s.e. over seeds)\ngrey = the reference run against itself with another seed", fontsize=10.5, loc="left"); ax2.legend(frameon=False, fontsize=8.2, loc="upper left")
fig.tight_layout(); f = "%s/fig_paths10_%s.png" % (a.out, a.scene); fig.savefig(f, dpi=160); print("wrote", f)
print("| arm | mean rel. distance to same-seed 14B 4-step (all frames) ↓ |"); print("|---|---|"); print("| 14B 4-step, different seed | %.3f |" % np.mean(tt))
for k, v in summ.items(): print("| %s | %.3f |" % (LAB[k], v))
