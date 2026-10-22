"""Figures for the velocity / score verification probe (specwm/velocity_probe.py). Writes paper/1005/meeting/04_velocity_verify/:
  fig_maps.png      example chunks: decoded target / draft frame, where the two velocities disagree (sigma 0.83 / 0.4), target denoising error map
  fig_dist.png      per signal and sigma: distribution on target chunks vs draft chunks (same context, same noise), AUC in the title
  fig_auc.png       AUC (target chunk vs draft chunk) against sigma for each signal
  fig_rank.png      among draft chunks only: Spearman of each signal with local_err (distance to the target's own chunk), and the node-0 agreement
  numbers.json, README.md
usage: CUDA_VISIBLE_DEVICES=g python scripts/velocity_viz.py"""
import os, sys, glob, json, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
S = "/home/hk28965/specwm"; P = S + "/lingbot-world-v2/specwm_out/velocity_probe"; O = S + "/paper/1005/meeting/04_velocity_verify"; os.makedirs(O, exist_ok=True)
R = [r for f in glob.glob(P + "/records_*.json") for r in json.load(open(f))]
SIG = sorted({float(k.split("_s")[1]) for r in R for k in r if k.startswith("T_s")}, reverse=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3dd", "axes.axisbelow": True})
CT, CD = "#2a78d6", "#eb6834"; CS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SIGNALS = [("vdis", "velocity disagreement ‖v_T−v_D‖²/‖v_T‖²", +1), ("eT", "target denoising error e_T", +1), ("ctr", "contrast log(e_T/e_D)", +1)]   # all: lower = more target-like
out = dict(n_pairs=len(R), sigmas=SIG, auc={}, spearman_local_err={})
# ---- AUC target vs draft chunk (score oriented so that larger = more target-like)
fig, axs = plt.subplots(len(SIGNALS), len(SIG), figsize=(3.0 * len(SIG), 2.4 * len(SIGNALS)), dpi=170)
for i, (k, nm, sgn) in enumerate(SIGNALS):
    out["auc"][k] = {}
    for j, s in enumerate(SIG):
        t = np.array([r["T_s%.2f" % s][k] for r in R]); d = np.array([r["D_s%.2f" % s][k] for r in R])
        auc = roc_auc_score([1] * len(t) + [0] * len(d), np.concatenate([t, d]) * (-sgn)); out["auc"][k][s] = round(float(auc), 3)
        ax = axs[i, j]; lo, hi = np.percentile(np.concatenate([t, d]), [1, 99]); bins = np.linspace(lo, hi, 40)
        ax.hist(t, bins, color=CT, alpha=0.55, label="target chunk"); ax.hist(d, bins, color=CD, alpha=0.55, label="draft chunk")
        ax.set_title("%s\nσ=%.2f  AUC %.2f" % (k, s, auc), fontsize=9); ax.tick_params(labelsize=7)
        if i == 0 and j == 0: ax.legend(frameon=False, fontsize=7)
fig.suptitle("Signal on the target's chunk vs the draft's chunk (same context, same noise); AUC: larger = more target-like", fontsize=10, x=0.01, ha="left"); fig.tight_layout(); fig.savefig(O + "/fig_dist.png"); plt.close(fig)
fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
for (k, nm, _), col in zip(SIGNALS, CS):
    ax.plot(SIG, [out["auc"][k][s] for s in SIG], "-o", color=col, lw=2, ms=6, label=nm)
ax.axhline(0.5, color="#888", lw=1, ls=":"); ax.set_xlabel("re-noising level σ (1 = pure noise)"); ax.set_ylabel("AUC target vs draft chunk ↑"); ax.set_ylim(0.4, 1.02); ax.invert_xaxis()
ax.set_title("Does the signal tell the target's chunk from the draft's?", fontsize=10, loc="left"); ax.legend(frameon=False, fontsize=8, loc="lower left"); fig.tight_layout(); fig.savefig(O + "/fig_auc.png"); plt.close(fig)
# ---- ranking draft chunks by closeness to the target's own chunk
le = np.array([r["local_err"] for r in R]); d0 = np.array([r["d0"] for r in R])
out["spearman_local_err"]["d0 (node-0 agreement, sigma 1)"] = round(float(spearmanr(d0, le)[0]), 3)
fig, axs = plt.subplots(1, 2, figsize=(10.0, 3.6), dpi=200)
for (k, nm, _), col in zip(SIGNALS, CS):
    rho = [float(spearmanr([r["D_s%.2f" % s][k] for r in R], le)[0]) for s in SIG]; out["spearman_local_err"][k] = dict(zip(SIG, [round(x, 3) for x in rho]))
    axs[0].plot(SIG, rho, "-o", color=col, lw=2, ms=6, label=k)
axs[0].axhline(out["spearman_local_err"]["d0 (node-0 agreement, sigma 1)"], color="#4a3aa7", lw=2, ls="--", label="d0 node-0 agreement")
axs[0].axhline(0, color="#888", lw=1, ls=":"); axs[0].invert_xaxis(); axs[0].set_xlabel("σ"); axs[0].set_ylabel("Spearman with local error"); axs[0].legend(frameon=False, fontsize=8)
fig.suptitle("Among draft chunks: does the signal rank them by their distance to the target's own chunk?", fontsize=10, x=0.01, ha="left")
best = max(((k, s) for k, _, _ in SIGNALS for s in SIG), key=lambda ks: abs(out["spearman_local_err"][ks[0]][ks[1]]))
x = np.array([r["D_s%.2f" % best[1]][best[0]] for r in R])
axs[1].scatter(d0, le, s=10, color="#4a3aa7", alpha=0.6, label="d0 (ρ %.2f)" % out["spearman_local_err"]["d0 (node-0 agreement, sigma 1)"])
ax2 = axs[1].twiny(); ax2.scatter(x, le, s=10, color=CS[[k for k, _, _ in SIGNALS].index(best[0])], alpha=0.5, label="%s σ%.2f (ρ %.2f)" % (best[0], best[1], out["spearman_local_err"][best[0]][best[1]]))
axs[1].set_xlabel("d0"); ax2.set_xlabel("%s at σ %.2f" % best); axs[1].set_ylabel("local error ‖x_D − x_T‖²/‖x_T‖²")
h1, l1 = axs[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); axs[1].legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(O + "/fig_rank.png"); plt.close(fig)
# ---- maps (decode one latent frame per candidate)
sys.path.insert(0, S + "/lingbot-world-v2")
from wan.modules.vae2_1 import Wan2_1_VAE
vae = Wan2_1_VAE(vae_pth=S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast/Wan2.1_VAE.pth", device="cuda")
import re
MR = re.compile(r"maps_(.+)_s(\d+)_c(\d+)_([TD])_s([\d.]+)\.pt$")
cases = sorted({(m.group(1), int(m.group(2)), int(m.group(3))) for m in (MR.search(os.path.basename(f)) for f in glob.glob(P + "/maps_*.pt")) if m})
cases = [c for c in cases if c[2] in (3, 4)][:4]
fig, axs = plt.subplots(len(cases), 6, figsize=(16, 2.35 * len(cases)), dpi=150)
def frame(lat):
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16): im = vae.decode([lat.float().cuda()])[0].float().clamp(-1, 1)
    return ((im[:, -1] + 1) / 2).permute(1, 2, 0).cpu().numpy()
for i, (task, seed, c) in enumerate(cases):
    f = lambda n, s: torch.load("%s/maps_%s_s%d_c%d_%s_s%.2f.pt" % (P, task, seed, c, n, s))
    T1, D1, D3, T3 = f("T", SIG[0]), f("D", SIG[0]), f("D", SIG[2]), f("T", SIG[2])
    vm = lambda m: m.float()[-1].numpy()
    panels = [(frame(T1["x"]), "target chunk"), (frame(D1["x"]), "draft chunk (same noise)"), (vm(D1["vdis"]), "|v_T − v_D|² at σ %.2f (draft chunk)" % SIG[0]),
              (vm(D3["vdis"]), "|v_T − v_D|² at σ %.2f" % SIG[2]), (vm(D3["eT"]), "target denoising error, draft chunk σ %.2f" % SIG[2]), (vm(T3["eT"]), "same, target chunk σ %.2f" % SIG[2])]
    hi = max(float(np.percentile(p[0], 99)) for p in panels[2:])
    for j, (img, title) in enumerate(panels):
        ax = axs[i, j]; ax.axis("off")
        ax.imshow(img) if j < 2 else ax.imshow(img, cmap="magma", vmin=0, vmax=hi)
        if i == 0: ax.set_title(title, fontsize=8)
    axs[i, 0].text(-0.05, 0.5, "%s s%d c%d" % (task.split("_")[0], seed, c), transform=axs[i, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8)
fig.tight_layout(); fig.savefig(O + "/fig_maps.png"); plt.close(fig)
json.dump(out, open(O + "/numbers.json", "w"), indent=1); print(json.dumps(out, indent=1))
