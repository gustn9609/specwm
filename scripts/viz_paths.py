"""Two figures for the early-exit / draft-target story.
(1) fig_paths_<scene>_s<seed>.png : the rollout as a path. Left: per-frame VAE latents of several arms projected with ONE shared PCA (fit on all arms'
    frames), drawn as time-coloured paths (start = circle, chunk boundaries = ticks). Right: latent distance of each arm's frame to the dense
    4-step target frame at the same time (chaotic rollouts, so this is coarse drift, not fidelity), chunk index on x.
(2) fig_cost_closeness.png : per-token gap to the 14B 4-step output (same noise, same context; sigma units) vs 14B denoising steps per token, for
    draft-only, uniform 1/2/3/4-step, early exit (learned / oracle, sparse / dense), with the 14B's own different-noise variability as a band.
usage: python scripts/viz_paths.py [--scene sandcastle_wave_impact --seed 4321]"""
import argparse, json, glob, os, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
ap = argparse.ArgumentParser(); ap.add_argument("--scene", default="sandcastle_wave_impact"); ap.add_argument("--seed", type=int, default=4321); a = ap.parse_args()
S = "/home/hk28965/specwm"; O = S + "/lingbot-world-v2/specwm_out"; P = S + "/paper/1006_early_exit"; os.makedirs(P, exist_ok=True)
C = dict(target="#1f1f1f", draft="#e07b00", uniform2="#2a7fd4", uniform3="#2aa198", ours="#b03aa6")   # categorical, fixed order
# ---------------- figure 1: paths
ARMS = [("target (14B, 4-step)", "target", O + "/early_cmp/%s/uniform4_s%d/lat.pt"), ("draft only (1.3B)", "draft", O + "/pipe_ff3/%s/draft_1p3b_s%d/lat.pt"),
        ("14B uniform 2-step", "uniform2", O + "/early_cmp/%s/uniform2_s%d/lat.pt"), ("14B uniform 3-step", "uniform3", O + "/early_cmp/%s/uniform3_s%d/lat.pt"),
        ("ours: early exit (2.3 steps)", "ours", O + "/early_cmp/%s/early0.24_flash_s%d/lat.pt")]
L = {}
for lab, key, pat in ARMS:
    p = pat % (a.scene, a.seed)
    if os.path.exists(p): L[key] = (lab, torch.load(p).float())   # [16, F, H, W]
F = min(v[1].shape[1] for v in L.values()); feats = {k: v[1][:, :F].permute(1, 0, 2, 3).reshape(F, -1).numpy() for k, v in L.items()}
pca = PCA(2, random_state=0).fit(np.concatenate(list(feats.values()))); Z = {k: pca.transform(f) for k, f in feats.items()}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw=dict(width_ratios=[1.05, 1])); fig.patch.set_facecolor("white")
for ax in (ax1, ax2): ax.set_facecolor("white"); [s.set_visible(False) for s in (ax.spines["top"], ax.spines["right"])]; ax.grid(True, color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
order = [k for _, k, _ in ARMS if k in Z]
for k in order:
    z = Z[k]; lab = L[k][0]; col = C[k]; t = np.linspace(0.25, 1.0, len(z) - 1)
    seg = np.stack([z[:-1], z[1:]], 1); lc = LineCollection(seg, colors=[matplotlib.colors.to_rgba(col, al) for al in t], lw=2.4 if k in ("ours", "target") else 1.8, zorder=3 if k == "ours" else 2)
    ax1.add_collection(lc); ax1.scatter(z[::12, 0], z[::12, 1], s=10, color=col, zorder=4, lw=0)   # chunk boundaries
    ax1.scatter(z[-1, 0], z[-1, 1], s=70, color=col, marker="X", zorder=5, lw=0); ax1.plot([], [], color=col, lw=2.2, label=lab)
ax1.scatter(Z[order[0]][0, 0], Z[order[0]][0, 1], s=120, facecolor="white", edgecolor="#1f1f1f", lw=1.8, zorder=6); ax1.annotate("start (shared first frame)", Z[order[0]][0], xytext=(8, 8), textcoords="offset points", fontsize=9, color="#444")
ax1.autoscale(); ax1.set_xlabel("PC 1 of frame latents (shared PCA)"); ax1.set_ylabel("PC 2"); ax1.set_title("Rollout paths in latent space — %s, seed %d\n(fading = earlier, dots = chunk boundaries, X = last frame)" % (a.scene.replace("_", " "), a.seed), fontsize=10.5, loc="left")
ax1.legend(frameon=False, fontsize=9, loc="best")
tg = feats["target"]
for k in order:
    if k == "target": continue
    d = np.linalg.norm(feats[k] - tg, axis=1) / np.linalg.norm(tg, axis=1); x = np.arange(len(d)) / 12.0
    ax2.plot(x, d, color=C[k], lw=2.4 if k == "ours" else 1.6, label=L[k][0], zorder=3 if k == "ours" else 2)
for c in range(1, F // 12 + 1): ax2.axvline(c, color="#eeeeee", lw=0.8, zorder=1)
ax2.set_xlabel("chunk index (12 frames each; 16 fps → 0.75 s)"); ax2.set_ylabel("relative latent distance to the 4-step target frame ↓"); ax2.set_ylim(bottom=0)
ax2.set_title("Distance to the 4-step target frame over time\n(coarse: rollouts are chaotic, all arms diverge)", fontsize=10.5, loc="left"); ax2.legend(frameon=False, fontsize=9)
fig.tight_layout(); f1 = "%s/paper/1012/fig_paths_%s_s%d.png" % (S, a.scene, a.seed); fig.savefig(f1, dpi=160); print("wrote", f1)
# ---------------- figure 2: cost vs closeness (same-noise, same-context token gaps)
def err_of(arm):
    Ls = [json.load(open(p)) for p in glob.glob(O + "/early_cmp/*/%s_s[49]*/loop.json" % arm)]
    e = [h[4] for l in Ls for h in l["per_chunk_commit_hist"] if len(h) > 4]; return (np.mean([l["mean_steps"] for l in Ls]), np.mean(e)) if e else None
gp = json.load(open(O + "/gap_probe/records_14b.json")); uni = {k: np.mean([r["mean"][str(k)] for r in gp]) for k in (1, 2, 3)}
cal = json.load(open(O + "/noise_calib/records.json")); tt = np.mean([r["mean_TT"] for r in cal]); dt = np.mean([r["mean_DT"] for r in cal])
pts = [("draft only (1.3B, 4 steps)", 0.0, dt, C["draft"], "o"), ("14B 1-step", 1, uni[1], C["uniform2"], "s"), ("14B 2-step", 2, uni[2], C["uniform2"], "s"), ("14B 3-step", 3, uni[3], C["uniform3"], "s"), ("14B 4-step (reference)", 4, 0.0, C["target"], "s")]
for arm, lab, mk in (("diag_learned_sparse_flash", "ours: early exit, learned, K/V reuse", "D"), ("diag_learned_sparse_min2", "ours: no exit before step 2", "D"), ("diag_learned_sparse_min3", "ours: no exit before step 3", "D"), ("diag_learned_sparse_0.24_min2_refresh", "ours: min-exit 2 + last-step refresh", "D"),
                     ("diag_oracle_sparse_flash", "oracle mask, K/V reuse", "^"), ("diag_oracle_dense_0.24", "oracle mask, dense recompute (no speed-up)", "v"), ("diag_learned_dense_0.24", "learned mask, dense recompute", "v")):
    r = err_of(arm)
    if r: pts.append((lab, r[0], r[1], C["ours"] if lab.startswith("ours") else "#7a7a7a", mk))
fig, ax = plt.subplots(figsize=(8.2, 5.6)); fig.patch.set_facecolor("white"); ax.set_facecolor("white"); [s.set_visible(False) for s in (ax.spines["top"], ax.spines["right"])]; ax.grid(True, color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
ax.axhspan(0, tt, color="#f3f3f3", zorder=0); ax.axhline(tt, color="#9a9a9a", lw=1, ls="--", zorder=1); ax.text(4.02, tt + 0.008, "14B vs 14B with a different noise (%.2f)" % tt, fontsize=8.5, color="#666", ha="right")
xs = [p[1] for p in pts if p[0].startswith("14B")]; ys = [p[2] for p in pts if p[0].startswith("14B")]; ax.plot(xs, ys, color="#bbbbbb", lw=1.2, zorder=1)
OFF = {"ours: no exit before step 2": (8, 10), "ours: no exit before step 3": (10, 2), "ours: min-exit 2 + last-step refresh": (-150, -22), "14B 3-step": (6, -18), "14B 2-step": (-62, 6), "oracle mask, K/V reuse": (-60, 16), "ours: early exit, learned, K/V reuse": (10, 10), "learned mask, dense recompute": (-120, -22), "oracle mask, dense recompute (no speed-up)": (-10, -34), "14B 4-step (reference)": (-150, 8)}
for lab, x, y, col, mk in pts:
    ax.scatter(x, y, s=95 if lab.startswith("ours") else 60, color=col, marker=mk, zorder=4, lw=0 if mk != "v" else 0.5, edgecolor="white")
    ax.annotate(lab, (x, y), xytext=OFF.get(lab, (7, 4)), textcoords="offset points", fontsize=8.5, color=col if col != "#7a7a7a" else "#555", fontweight="bold" if lab.startswith("ours") else "normal")
ax.set_xlabel("14B denoising forwards per token (draft-only = 0; its 1.3B cost ≈ 0.15 of a 14B forward)"); ax.set_ylabel("token gap to the 14B 4-step output ↓  (same noise & context, σ units)")
ax.set_title("Cost vs closeness to the target: where early exit sits between the dense samplers", fontsize=11, loc="left"); ax.set_xlim(-0.2, 4.25); ax.set_ylim(-0.01, max(dt, 0.65) + 0.03)
fig.tight_layout(); f2 = S + "/paper/1012/fig_cost_closeness.png"; fig.savefig(f2, dpi=160); print("wrote", f2)
