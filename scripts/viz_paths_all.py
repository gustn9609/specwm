"""Rollout-path figure across model families (paper/1012/fig_paths_all_<scene>.png): teacher 14B, student 1.3B, ours (704 / 832 fused + partial
commit), SDVG with actions (tau -0.7 / -2.2), Matrix-Game 2.0, Waypoint-1.5-1B; 10 seeds each.
The game models have their own latent spaces, so EVERY arm (teacher included) is put through the same round trip: decoded video (mp4) ->
undo the model's own framing (Waypoint stretches the start image to 1024x512: un-stretch to the source aspect; MG2 resize-crops to 352x640) ->
centre-crop to the teacher's aspect -> resize to half the teacher grid -> Wan2.1 VAE encode. Re-encoded latents are cached next to each video
(reenc_<H>x<W>.pt). Left: one shared PCA over all arms and seeds (thin = seeds, thick = per-frame mean). Right: relative latent distance to the
same-seed teacher frame (mean +- s.e.), grey dashed = teacher vs teacher with a different seed (the reference's own spread).
usage: CUDA_VISIBLE_DEVICES=g python scripts/viz_paths_all.py --scene easy_lake [--seeds ...]"""
import argparse, os, sys, json, numpy as np, torch, cv2, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
S = "/home/hk28965/specwm"; O = S + "/lingbot-world-v2/specwm_out"; sys.path.insert(0, S + "/lingbot-world-v2")
ap = argparse.ArgumentParser(); ap.add_argument("--scene", default="easy_lake"); ap.add_argument("--seeds", default="1234,4321,9999,11,22,33,44,55,66,77"); ap.add_argument("--out", default=S + "/paper/1012")
ap.add_argument("--frames", type=int, default=153); a = ap.parse_args(); SEEDS = [int(s) for s in a.seeds.split(",")]; sc = a.scene
T = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
ARMS = [  # (label, key, colour, path(seed), source kind)
    ("teacher: 14B 4-step", "teacher", "#1f1f1f", lambda s: O + "/early_cmp/%s/uniform4_s%d/video.mp4" % (sc, s), "lingbot"),
    ("student: 1.3B 4-step (edge only)", "student", "#e07b00", lambda s: O + "/pipe_ff3/%s/draft_1p3b_s%d/video.mp4" % (sc, s), "lingbot"),
    ("SDVG + actions, τ −0.7 (1.3B draft, 14B target)", "sdvg07", "#8d6e63", lambda s: O + "/sdvg_paths/%s/ir_-0.7_s%d/video.mp4" % (sc, s), "lingbot"),
    ("SDVG + actions, τ −2.2", "sdvg22", "#c49a6c", lambda s: O + "/sdvg_paths/%s/ir_-2.2_s%d/video.mp4" % (sc, s), "lingbot"),
    ("Matrix-Game 2.0 (edge SOTA)", "mg2", "#2e7d32", lambda s: O + "/mg2/seeds/%s_s%d/video.mp4" % (sc, s), "mg2"),
    ("Waypoint-1.5-1B (edge SOTA)", "wp", "#00897b", lambda s: O + "/waypoint/seeds/%s_s%d/output.mp4" % (sc, s), "wp"),
    ("ours: 832 fused + partial commit (9.7 fps)", "ours832", "#7b4bd1", lambda s: O + "/fused/%s/pc1_832_s%d/cloud/spec.mp4" % (sc, s), "lingbot"),
    ("ours: 704 fused + partial commit (15.8 fps)", "ours704", "#d81b60", lambda s: O + "/fused/%s/pc1_704_s%d/cloud/spec.mp4" % (sc, s), "lingbot")]
OURS = ("ours704", "ours832")
def read(p):
    cap = cv2.VideoCapture(p); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return fr
src = cv2.imread(T[sc]["image"]); SA = src.shape[1] / src.shape[0]   # source image aspect (w / h)
ref = read(ARMS[0][3](SEEDS[0])); TH, TW = ref[0].shape[:2]; TA = TW / TH; H2, W2 = (TH // 2) // 8 * 8, (TW // 2) // 8 * 8
def frame_to_grid(f, kind):
    h, w = f.shape[:2]
    if kind == "wp": f = cv2.resize(f, (int(round(h * SA)), h), interpolation=cv2.INTER_AREA); h, w = f.shape[:2]   # undo Waypoint's stretch
    if w / h > TA: nw = int(round(h * TA)); x0 = (w - nw) // 2; f = f[:, x0:x0 + nw]
    else: nh = int(round(w / TA)); y0 = (h - nh) // 2; f = f[y0:y0 + nh]
    return cv2.resize(f, (W2, H2), interpolation=cv2.INTER_AREA)
VAE = {}
def encode(p, kind):
    c = os.path.join(os.path.dirname(p), "reenc_%dx%d.pt" % (H2, W2))
    if os.path.exists(c): return torch.load(c).float()
    if "m" not in VAE:
        from wan.modules.vae2_1 import Wan2_1_VAE
        VAE["m"] = Wan2_1_VAE(vae_pth=S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast/Wan2.1_VAE.pth", device="cuda")
    fr = read(p)[:a.frames]
    if len(fr) < a.frames: return None
    x = torch.from_numpy(np.stack([frame_to_grid(f, kind) for f in fr])).permute(3, 0, 1, 2).float().div(127.5).sub(1).cuda()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16): z = VAE["m"].encode([x])[0].float().cpu()
    torch.save(z.half(), c); return z
L = {}
for lab, key, col, pf, kind in ARMS:
    for s in SEEDS:
        p = pf(s)
        if os.path.exists(p):
            z = encode(p, kind)
            if z is not None: L.setdefault(key, {})[s] = z.permute(1, 0, 2, 3).reshape(z.shape[1], -1).numpy()
print(sc, {k: len(d) for k, d in L.items()}, "grid", H2, W2, flush=True)
F = min(v.shape[0] for d in L.values() for v in d.values())
for d in L.values():
    for s in d: d[s] = d[s][:F]
LAB = {k: lab for lab, k, _, _, _ in ARMS}; COL = {k: c for _, k, c, _, _ in ARMS}; order = [k for _, k, _, _, _ in ARMS if k in L]
pca = PCA(2, random_state=0).fit(np.concatenate([v for d in L.values() for v in d.values()])); Z = {k: {s: pca.transform(v) for s, v in d.items()} for k, d in L.items()}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw=dict(width_ratios=[1.05, 1])); fig.patch.set_facecolor("white")
for ax in (ax1, ax2): ax.set_facecolor("white"); [ax.spines[q].set_visible(False) for q in ("top", "right")]; ax.grid(True, color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
for k in order:
    big = k in OURS or k == "teacher"
    for s, z in Z[k].items(): ax1.plot(z[:, 0], z[:, 1], color=COL[k], lw=0.5, alpha=0.10, zorder=2)
    m = np.mean(np.stack(list(Z[k].values())), 0); m = np.stack([np.convolve(np.pad(m[:, j], 1, mode="edge"), np.ones(3) / 3, mode="valid") for j in range(2)], 1)   # 3-frame moving average (display only)
    ax1.plot(m[:, 0], m[:, 1], color=COL[k], lw=2.8 if big else 1.9, zorder=4 if big else 3, label="%s  (n=%d)" % (LAB[k], len(Z[k])))
    ax1.scatter(m[::3, 0], m[::3, 1], s=11, color=COL[k], zorder=5, lw=0); ax1.scatter(m[-1, 0], m[-1, 1], s=80, color=COL[k], marker="X", zorder=6, lw=0)
m0 = np.mean(np.stack(list(Z["teacher"].values())), 0)[0]; ax1.scatter(*m0, s=120, facecolor="white", edgecolor="#1f1f1f", lw=1.8, zorder=7); ax1.annotate("start (shared first frame)", m0, xytext=(8, 8), textcoords="offset points", fontsize=9, color="#444")
ax1.set_xlabel("PC 1 of frame latents (shared PCA over all arms and seeds)"); ax1.set_ylabel("PC 2"); ax1.legend(frameon=False, fontsize=8, loc="best")
ax1.set_title("Rollout paths in latent space — %s\n(all arms re-encoded with the Wan2.1 VAE on one grid; thin = seeds, thick = seed mean, 3-frame moving average; X = last frame)" % sc.replace("_", " "), fontsize=10.5, loc="left")
def rel(x, t): return np.linalg.norm(x - t, axis=1) / np.linalg.norm(t, axis=1)
TT = L["teacher"]; xs = np.arange(F) / 3.0
def band(D, col, lab, lw, ls="-", z=3):
    D = np.stack(D); m = D.mean(0); se = D.std(0, ddof=1) / np.sqrt(len(D)) if len(D) > 1 else 0 * m
    ax2.plot(xs, m, color=col, lw=lw, ls=ls, label="%s  (n=%d)" % (lab, len(D)), zorder=z); ax2.fill_between(xs, m - se, m + se, color=col, alpha=0.14, lw=0, zorder=z - 1); return m
summ = {"teacher_spread": float(band([rel(TT[s2], TT[s1]) for s1 in TT for s2 in TT if s1 != s2], "#8a8a8a", "teacher vs teacher, different seed (reference spread)", 1.6, "--", 2).mean())}
for k in order:
    if k == "teacher": continue
    D = [rel(L[k][s], TT[s]) for s in L[k] if s in TT]
    if D: summ[k] = float(band(D, COL[k], LAB[k], 2.6 if k in OURS else 1.6, z=4 if k in OURS else 3).mean())
for c in range(1, F // 3 + 1): ax2.axvline(c, color="#eeeeee", lw=0.8, zorder=1)
ax2.set_xlabel("chunk index (3 latent frames = 12 video frames = 0.75 s)"); ax2.set_ylabel("relative latent distance to the same-seed teacher frame ↓"); ax2.set_ylim(bottom=0)
ax2.set_title("Distance to the teacher over time (mean ± s.e. over seeds)\nours-832 starts each chunk from the teacher's noise; the other arms use other noise streams, so compare them with the grey band", fontsize=10.5, loc="left"); ax2.legend(frameon=False, fontsize=7.6, loc="lower right")
fig.tight_layout(); os.makedirs(a.out, exist_ok=True); fn = a.out + "/fig_paths_all_%s.png" % sc; fig.savefig(fn, dpi=150); print("[FIG]", fn, json.dumps({k: round(v, 3) for k, v in summ.items()}), flush=True)
json.dump(dict(scene=sc, n={k: len(d) for k, d in L.items()}, mean_rel_dist=summ, grid=[H2, W2]), open(a.out + "/fig_paths_all_%s.json" % sc, "w"), indent=1)
