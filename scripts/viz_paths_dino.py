"""Rollout paths in DINOv2 feature space (replaces the latent-PCA panel of viz_paths.py).
Per frame: DINOv2-large CLS embedding (what the scene is) of the decoded video. Left: one shared PCA over all arms' frames, time-coloured paths.
Right: cosine similarity of each arm's frame to (a) the first frame (does the rollout stay in the starting world?) and (b) the 4-step target frame at
the same time. usage: CUDA_VISIBLE_DEVICES=g python scripts/viz_paths_dino.py [--scene sandcastle_wave_impact --seed 4321]"""
import argparse, os, numpy as np, torch, matplotlib, imageio.v3 as iio
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor
from transformers.models.dinov2.modeling_dinov2 import Dinov2Model as AutoModel
ap = argparse.ArgumentParser(); ap.add_argument("--scene", default="sandcastle_wave_impact"); ap.add_argument("--seed", type=int, default=4321); ap.add_argument("--umap", action="store_true"); a = ap.parse_args()
S = "/home/hk28965/specwm"; O = S + "/lingbot-world-v2/specwm_out"; P = S + "/paper/1006_early_exit"
C = dict(target="#1f1f1f", draft="#e07b00", uniform2="#2a7fd4", uniform3="#2aa198", ours="#b03aa6")
ARMS = [("target (14B, 4-step)", "target", O + "/early_cmp/%s/uniform4_s%d/video.mp4"), ("draft only (1.3B)", "draft", O + "/pipe_ff3/%s/draft_1p3b_s%d/video.mp4"),
        ("14B uniform 2-step", "uniform2", O + "/early_cmp/%s/uniform2_s%d/video.mp4"), ("14B uniform 3-step", "uniform3", O + "/early_cmp/%s/uniform3_s%d/video.mp4"),
        ("ours: early exit (2.3 steps)", "ours", O + "/early_cmp/%s/early0.24_flash_s%d/video.mp4")]
proc = AutoImageProcessor.from_pretrained(S + "/hf/dinov2-large"); m = AutoModel.from_pretrained(S + "/hf/dinov2-large").cuda().eval().half()
@torch.no_grad()
def embed(p):
    v = iio.imread(p, plugin="pyav"); out = []
    for i in range(0, len(v), 32):
        x = proc(images=list(v[i:i + 32]), return_tensors="pt")["pixel_values"].cuda().half(); h = m(pixel_values=x).last_hidden_state
        out.append(torch.nn.functional.normalize(h[:, 0].float(), dim=-1).cpu())   # CLS
    return torch.cat(out).numpy()
E = {}
for lab, key, pat in ARMS:
    p = pat % (a.scene, a.seed)
    if os.path.exists(p): E[key] = (lab, embed(p)); print(key, E[key][1].shape, flush=True)
F = min(v[1].shape[0] for v in E.values()); feats = {k: v[1][:F] for k, v in E.items()}
allf = np.concatenate(list(feats.values()))
if a.umap:
    import umap; red = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=0).fit(allf); Z = {k: red.transform(f) for k, f in feats.items()}; axl = ("UMAP 1", "UMAP 2")
else:
    red = PCA(2, random_state=0).fit(allf); Z = {k: red.transform(f) for k, f in feats.items()}; axl = ("PC 1 of DINOv2 frame embeddings (shared)", "PC 2")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.6), gridspec_kw=dict(width_ratios=[1.15, 1, 1])); fig.patch.set_facecolor("white")
for ax in (ax1, ax2, ax3): ax.set_facecolor("white"); [s.set_visible(False) for s in (ax.spines["top"], ax.spines["right"])]; ax.grid(True, color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
order = [k for _, k, _ in ARMS if k in Z]
for k in order:
    z = Z[k]; col = C[k]; t = np.linspace(0.2, 1.0, len(z) - 1); seg = np.stack([z[:-1], z[1:]], 1)
    ax1.add_collection(LineCollection(seg, colors=[matplotlib.colors.to_rgba(col, al) for al in t], lw=2.6 if k in ("ours", "target") else 1.8, zorder=3 if k == "ours" else 2))
    ax1.scatter(z[9::12, 0], z[9::12, 1], s=11, color=col, zorder=4, lw=0); ax1.scatter(z[-1, 0], z[-1, 1], s=80, color=col, marker="X", zorder=5, lw=0); ax1.plot([], [], color=col, lw=2.2, label=E[k][0])
z0 = Z[order[0]][0]; ax1.scatter(z0[0], z0[1], s=130, facecolor="white", edgecolor="#1f1f1f", lw=1.8, zorder=6); ax1.annotate("start (shared first frame)", z0, xytext=(8, 8), textcoords="offset points", fontsize=9, color="#444")
ax1.autoscale(); ax1.set_xlabel(axl[0]); ax1.set_ylabel(axl[1]); ax1.set_title("Where each rollout goes (DINOv2 space) — %s, seed %d\n(fading = earlier, dots = chunk ends, X = last frame)" % (a.scene.replace("_", " "), a.seed), fontsize=10.5, loc="left"); ax1.legend(frameon=False, fontsize=9)
x = np.arange(F) / 12.0
for k in order:
    f = feats[k]; s0 = f @ f[0]; ax2.plot(x, s0, color=C[k], lw=2.4 if k == "ours" else 1.6, label=E[k][0], zorder=3 if k == "ours" else 2)
    if k != "target": ax3.plot(x, (f * feats["target"]).sum(1), color=C[k], lw=2.4 if k == "ours" else 1.6, label=E[k][0], zorder=3 if k == "ours" else 2)
for ax in (ax2, ax3):
    for c in range(1, F // 12 + 1): ax.axvline(c, color="#eeeeee", lw=0.8, zorder=1)
    ax.set_xlabel("chunk index (12 frames = 0.75 s)"); ax.set_ylim(0, 1.02); ax.legend(frameon=False, fontsize=8.5, loc="lower left")
ax2.set_ylabel("cosine similarity to the FIRST frame ↑"); ax2.set_title("Does the rollout stay in the starting world?", fontsize=10.5, loc="left")
ax3.set_ylabel("cosine similarity to the 4-step target frame ↑"); ax3.set_title("Agreement with the target frame at the same time\n(chaotic rollouts: coarse)", fontsize=10.5, loc="left")
fig.tight_layout(); f1 = "%s/fig_paths_dino_%s_s%d%s.png" % (P, a.scene, a.seed, "_umap" if a.umap else ""); fig.savefig(f1, dpi=160); print("wrote", f1)
np.savez("%s/dino_feats_%s_s%d.npz" % (P, a.scene, a.seed), **{k: v for k, v in feats.items()})
