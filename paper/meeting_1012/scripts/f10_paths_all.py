"""f10_paths_all -- rollout-path figures across baselines (meeting 10/12), reviewer-polished version.

Data and core computation are unchanged from scripts/viz_paths_all.py (the original is untouched):
  arms     teacher 14B 4-step, student 1.3B 4-step, SDVG + actions (tau -0.7 / -2.2), Matrix-Game 2.0, Waypoint-1.5-1B,
           ours 832 / 704 fused + partial commit; seeds 1234,4321,9999,11,22,33,44,55,66,77.
  latents  every arm (teacher included) decoded mp4 -> undo the model's own framing (Waypoint's 2:1 stretch) -> centre-crop to the
           teacher's aspect -> resize to half the teacher grid -> Wan2.1 VAE encode, cached next to each video as reenc_<H>x<W>.pt.
  distance rel = |z_arm - z_teacher| / |z_teacher| per latent frame against the teacher clip with the same seed index;
           reference = teacher vs teacher (all ordered pairs of different teacher seeds); ratio = time-mean distance / time-mean reference.

What this version adds (reviewer items, 10/12):
  - per-seed time-mean distances, seed s.e., paired (same seed index) ours 704 - student differences, all written to
    meeting_1012/data/f10_paths_<scene>.json; the summary reads those (means cross-checked against paper/1012/fig_paths_all_<scene>.json);
  - the reference's uncertainty as a leave-one-teacher-seed-out jackknife (ordered teacher pairs are not independent);
  - SDVG draft-chunk counts ('accepted' in sdvg_paths/<scene>/ir_<tau>_s<seed>/loop.json) over the seeds that enter the distance;
  - PCA explained-variance fractions on the left-panel axes;
  - wording: 1.0 = as far as another teacher sample, < 1 is NOT better; ours 832 flagged as sharing the teacher's noise;
  - summary as a dot plot (one row per arm, scene dots with seed s.e., mean +- s.e. over scenes).

This version never loads the VAE by default: a missing reenc_*.pt cache is reported and the scene is skipped
(pass --allow-encode, with a GPU, to re-encode as the original script did).

usage: CUDA_VISIBLE_DEVICES="" venv/bin/python paper/meeting_1012/scripts/f10_paths_all.py [--scenes a,b] [--summary-only]
"""
import argparse, os, sys, json, textwrap
import numpy as np, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.transforms as mtransforms

S = "/home/hk28965/specwm"; O = S + "/lingbot-world-v2/specwm_out"; M = S + "/paper/meeting_1012"
sys.path.insert(0, S + "/lingbot-world-v2")
SCENES = ["sand_filling_up_jar", "sandcastle_wave_impact", "easy_lake", "easy_stonehenge_sunset", "easy_urban_rain", "pouring_water_into_cup"]
PRETTY = {"sand_filling_up_jar": "Sand filling up jar", "sandcastle_wave_impact": "Sandcastle wave impact", "easy_lake": "Easy lake",
          "easy_stonehenge_sunset": "Stonehenge sunset", "easy_urban_rain": "Urban rain", "pouring_water_into_cup": "Pouring water into cup"}
NICK = {"sand_filling_up_jar": "jar", "sandcastle_wave_impact": "sandcastle", "easy_lake": "lake", "easy_stonehenge_sunset": "Stonehenge",
        "easy_urban_rain": "urban rain", "pouring_water_into_cup": "pouring"}
ap = argparse.ArgumentParser()
ap.add_argument("--scenes", default=",".join(SCENES)); ap.add_argument("--seeds", default="1234,4321,9999,11,22,33,44,55,66,77")
ap.add_argument("--frames", type=int, default=153); ap.add_argument("--out", default=M + "/figs"); ap.add_argument("--data", default=M + "/data")
ap.add_argument("--src_json", default=S + "/paper/1012"); ap.add_argument("--summary-only", action="store_true")
ap.add_argument("--allow-encode", action="store_true", help="re-encode on a cache miss (needs the GPU + VAE); off by default")
a = ap.parse_args(); SEEDS = [int(s) for s in a.seeds.split(",")]
os.makedirs(a.out, exist_ok=True); os.makedirs(a.data, exist_ok=True)

# ---------------------------------------------------------------- style (fixed arm colours across all meeting figures)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11.5, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "svg.fonttype": "none", "figure.facecolor": "white", "axes.facecolor": "white", "hatch.linewidth": 0.6})
GRID = "#e6e6e6"; INK2 = "#444444"; FOOT = "#6b6b6b"; REF = "#8a8a8a"
NC = "shares the teacher's noise: not comparable"
# (legend label, key, colour, path(scene, seed), source kind, short name)
ARMS = [
    ("teacher: 14B, 4 steps", "teacher", "#1f1f1f", lambda sc, s: O + "/early_cmp/%s/uniform4_s%d/video.mp4" % (sc, s), "lingbot", "teacher"),
    ("student: 1.3B, 4 steps (edge only)", "student", "#e07b00", lambda sc, s: O + "/pipe_ff3/%s/draft_1p3b_s%d/video.mp4" % (sc, s), "lingbot", "student"),
    ("SDVG + actions, τ −0.7 (1.3B draft, 14B target)", "sdvg07", "#8d6e63", lambda sc, s: O + "/sdvg_paths/%s/ir_-0.7_s%d/video.mp4" % (sc, s), "lingbot", "SDVG τ−0.7"),
    ("SDVG + actions, τ −2.2", "sdvg22", "#c49a6c", lambda sc, s: O + "/sdvg_paths/%s/ir_-2.2_s%d/video.mp4" % (sc, s), "lingbot", "SDVG τ−2.2"),
    ("Matrix-Game 2.0 (edge SOTA)", "mg2", "#2e7d32", lambda sc, s: O + "/mg2/seeds/%s_s%d/video.mp4" % (sc, s), "mg2", "Matrix-Game 2.0"),
    ("Waypoint-1.5-1B (edge SOTA)", "wp", "#00897b", lambda sc, s: O + "/waypoint/seeds/%s_s%d/output.mp4" % (sc, s), "wp", "Waypoint"),
    ("ours 832 (%s)" % NC, "ours832", "#7b4bd1", lambda sc, s: O + "/fused/%s/pc1_832_s%d/cloud/spec.mp4" % (sc, s), "lingbot", "ours 832"),
    ("ours 704: fused + partial commit", "ours704", "#d81b60", lambda sc, s: O + "/fused/%s/pc1_704_s%d/cloud/spec.mp4" % (sc, s), "lingbot", "ours 704")]
OURS = ("ours704", "ours832"); KEY = OURS + ("teacher",)
LAB = {k: lab for lab, k, *_ in ARMS}; COL = {k: c for _, k, c, *_ in ARMS}; SHORT = {k: sh for _, k, _, _, _, sh in ARMS}
LS = {k: ("-." if k == "ours832" else "-") for k in COL}                      # ours 832 drawn dash-dot everywhere (shared noise)
SDVG_TAU = {"sdvg07": "-0.7", "sdvg22": "-2.2"}
VID_ROOTS = ("lingbot-world-v2/specwm_out/{early_cmp (teacher), pipe_ff3 (student), sdvg_paths, mg2/seeds, waypoint/seeds, fused (ours)}/<scene>…")
SCRIPT = "paper/meeting_1012/scripts/f10_paths_all.py"
CAVEAT = ("Caveat: this is distance from the teacher's samples, not quality. An independent teacher sample scores ≈1.0 by construction; values below 1 "
          "come from shared noise (ours 832) or from outputs that vary less than the teacher, and do not mean better.")


def style(ax):
    for q in ("top", "right"): ax.spines[q].set_visible(False)
    for q in ("left", "bottom"): ax.spines[q].set_color("#9a9a9a")
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(colors=INK2)


def se(x):
    x = np.asarray(x, float); return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def sg(x, d=2):
    """signed number with a typographic minus"""
    return ("%+.*f" % (d, x)).replace("-", "−")


def verdict(m, s, closer="closer", farther="farther"):
    return "within seed noise" if abs(m) < 2 * s else (closer if m < 0 else farther)


def sdvg_draft(sc, key, seeds):
    """Chunks taken from the 1.3B draft (of 13), mean over the given seeds, from SDVG's loop.json ('accepted')."""
    v = {}
    for s in seeds:
        p = "%s/sdvg_paths/%s/ir_%s_s%d/loop.json" % (O, sc, SDVG_TAU[key], s)
        if os.path.exists(p): j = json.load(open(p)); v[s] = (int(j["accepted"]), int(j["chunks"]))
    return v


def fig_text(fig, text, x, y, size, color, va="top", width_frac=0.976, wrap=None, **kw):
    """Left-aligned figure text; each paragraph ("\n"-separated) is re-wrapped until the widest line fits the canvas."""
    paras = text.split("\n"); n = wrap or max(len(p) for p in paras)
    t = fig.text(x, y, "", fontsize=size, ha="left", va=va, color=color, **kw)
    for _ in range(8):
        t.set_text("\n".join(l for p in paras for l in (textwrap.wrap(p, n) or [""])))
        fig.canvas.draw(); w = t.get_window_extent().width / (fig.get_figwidth() * fig.dpi)
        if w <= width_frac: break
        n = max(20, int(n * width_frac / w) - 2)
    return t


def fbox(fig, artist):
    fig.canvas.draw(); return artist.get_window_extent().transformed(fig.transFigure.inverted())


def layout(fig, gs, axes, title, sub, foot, legend_kw, pad=0.014, foot_wrap=178):
    """Stack: title / subtitle / panels / shared legend / footer, measured so nothing overlaps or clips."""
    tt = fig_text(fig, title, 0.012, 0.978, 13, "#1f1f1f")
    st = fig_text(fig, sub, 0.012, fbox(fig, tt).y0 - 0.008, 11, INK2)
    ft = fig_text(fig, foot, 0.012, 0.012, 9, FOOT, va="bottom", wrap=foot_wrap, linespacing=1.35)
    for ncol in range(legend_kw.pop("ncol", 4), 0, -1):
        lg = fig.legend(loc="lower left", bbox_to_anchor=(0.012, fbox(fig, ft).y1 + pad), ncol=ncol, frameon=False, **legend_kw)
        if fbox(fig, lg).x1 <= 0.992: break
        lg.remove()
    lo, hi = fbox(fig, lg).y1 + pad, fbox(fig, st).y0 - 0.02
    left, right, bottom, top = 0.07, 0.985, lo + 0.1, hi - 0.05
    for _ in range(5):
        if bottom >= top: break                            # shift the panel block until its tight boxes sit inside [lo, hi] x [0.008, 0.992]
        gs.update(left=left, right=right, bottom=bottom, top=top); fig.canvas.draw()
        bb = [ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()) for ax in axes]
        bottom += lo - min(b.y0 for b in bb); top -= max(b.y1 for b in bb) - hi
        left += 0.008 - min(b.x0 for b in bb); right -= max(b.x1 for b in bb) - 0.992
    if bottom >= top: raise RuntimeError("layout: no room left for the panels (footer/legend/title too tall)")
    gs.update(left=left, right=right, bottom=bottom, top=top)


# ---------------------------------------------------------------- per-scene figures (same data + core computation as viz_paths_all.py)
def read(p):
    cap = cv2.VideoCapture(p); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return fr


def scene_figure(sc, T):
    import torch
    from sklearn.decomposition import PCA
    src = cv2.imread(T[sc]["image"]); SA = src.shape[1] / src.shape[0]                     # source image aspect (w / h)
    ref = read(ARMS[0][3](sc, SEEDS[0])); TH, TW = ref[0].shape[:2]; TA = TW / TH; H2, W2 = (TH // 2) // 8 * 8, (TW // 2) // 8 * 8

    def frame_to_grid(f, kind):
        h, w = f.shape[:2]
        if kind == "wp": f = cv2.resize(f, (int(round(h * SA)), h), interpolation=cv2.INTER_AREA); h, w = f.shape[:2]   # undo Waypoint's stretch
        if w / h > TA: nw = int(round(h * TA)); x0 = (w - nw) // 2; f = f[:, x0:x0 + nw]
        else: nh = int(round(w / TA)); y0 = (h - nh) // 2; f = f[y0:y0 + nh]
        return cv2.resize(f, (W2, H2), interpolation=cv2.INTER_AREA)

    VAE, miss = {}, []
    def encode(p, kind):
        c = os.path.join(os.path.dirname(p), "reenc_%dx%d.pt" % (H2, W2))
        if os.path.exists(c): return torch.load(c).float()
        miss.append(c)
        if not a.allow_encode: return None                                                        # report, never force the VAE
        if "m" not in VAE:
            from wan.modules.vae2_1 import Wan2_1_VAE
            VAE["m"] = Wan2_1_VAE(vae_pth=S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast/Wan2.1_VAE.pth", device="cuda")
        fr = read(p)[:a.frames]
        if len(fr) < a.frames: return None
        x = torch.from_numpy(np.stack([frame_to_grid(f, kind) for f in fr])).permute(3, 0, 1, 2).float().div(127.5).sub(1).cuda()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16): z = VAE["m"].encode([x])[0].float().cpu()
        torch.save(z.half(), c); return z

    L = {}
    for lab, key, col, pf, kind, _ in ARMS:
        for s in SEEDS:
            p = pf(sc, s)
            if os.path.exists(p):
                z = encode(p, kind)
                if z is not None: L.setdefault(key, {})[s] = z.permute(1, 0, 2, 3).reshape(z.shape[1], -1).numpy()
    if miss and not a.allow_encode:
        print("[CACHE MISS] %s: %d re-encoded latents missing, scene skipped (would need the VAE on a GPU):" % (sc, len(miss)), flush=True)
        for c in miss: print("   ", c, flush=True)
        return None
    print(sc, {k: len(d) for k, d in L.items()}, "grid", H2, W2, flush=True)
    F = min(v.shape[0] for d in L.values() for v in d.values())
    for d in L.values():
        for s in d: d[s] = d[s][:F]
    order = [k for _, k, *_ in ARMS if k in L]
    pca = PCA(2, random_state=0).fit(np.concatenate([v for d in L.values() for v in d.values()]))
    evr = [float(v) for v in pca.explained_variance_ratio_]
    Z = {k: {s: pca.transform(v) for s, v in d.items()} for k, d in L.items()}

    # distances (identical to the original) + per-seed values and honest uncertainty
    def rel(x, t): return np.linalg.norm(x - t, axis=1) / np.linalg.norm(t, axis=1)
    TT = L["teacher"]; xs = np.arange(F) / 3.0; tk = sorted(TT)
    def spread(seeds): return np.stack([rel(TT[s2], TT[s1]) for s1 in seeds for s2 in seeds if s1 != s2])
    curves = {"teacher_spread": spread(tk)}
    for k in order:
        if k == "teacher": continue
        D = [rel(L[k][s], TT[s]) for s in L[k] if s in TT]
        if D: curves[k] = np.stack(D)
    summ = {k: float(D.mean(0).mean()) for k, D in curves.items()}
    ratio = {k: summ[k] / summ["teacher_spread"] for k in summ if k != "teacher_spread"}
    # leave-one-teacher-seed-out jackknife for the reference (ordered pairs share seeds, so they are not independent)
    nT = len(tk); jk = np.stack([spread([s for s in tk if s != i]).mean(0) for i in tk])          # nT x F
    jk_curve = np.sqrt((nT - 1) / nT * ((jk - jk.mean(0)) ** 2).sum(0))
    jk_t = jk.mean(1); jk_mean_se = float(np.sqrt((nT - 1) / nT * ((jk_t - jk_t.mean()) ** 2).sum()))
    used = {k: [s for s in SEEDS if s in L[k] and s in TT] for k in curves if k != "teacher_spread"}
    per_seed = {k: {s: float(rel(L[k][s], TT[s]).mean()) for s in used[k]} for k in used}
    rseed = {k: [v / summ["teacher_spread"] for v in per_seed[k].values()] for k in per_seed}
    ratio_se = {k: se(v) for k, v in rseed.items()}
    sh = [s for s in used["ours704"] if s in used["student"]]
    dd = [(per_seed["ours704"][s] - per_seed["student"][s]) / summ["teacher_spread"] for s in sh]
    paired = dict(mean=float(np.mean(dd)), se=se(dd), n=len(dd))
    draft = {k: sdvg_draft(sc, k, used[k]) for k in SDVG_TAU if k in used}
    draft_mean = {k: float(np.mean([v[0] for v in d.values()])) for k, d in draft.items() if d}
    nchunk = {v[1] for d in draft.values() for v in d.values()}; nchunk = nchunk.pop() if len(nchunk) == 1 else None

    # ---- layout: title / two panels / shared legend / footer
    fig = plt.figure(figsize=(12, 7.6), dpi=200)
    gs = fig.add_gridspec(1, 2, wspace=0.30, width_ratios=[1, 1.05])
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]); style(ax1); style(ax2)

    # left: rollout paths. Baselines recede (thin, lighter); teacher and ours are drawn last and thick.
    for k in sorted(order, key=lambda k: (k in KEY, k in OURS)):
        big = k in KEY
        for s, z in Z[k].items(): ax1.plot(z[:, 0], z[:, 1], color=COL[k], lw=0.45, alpha=0.08 if not big else 0.12, zorder=2)
        m = np.mean(np.stack(list(Z[k].values())), 0)
        m = np.stack([np.convolve(np.pad(m[:, j], 1, mode="edge"), np.ones(3) / 3, mode="valid") for j in range(2)], 1)   # display only
        ax1.plot(m[:, 0], m[:, 1], color=COL[k], lw=2.6 if big else 1.4, alpha=1.0 if big else 0.85, zorder=5 if big else 3,
                 solid_capstyle="round", ls=LS[k])
        if big: ax1.scatter(m[::3, 0], m[::3, 1], s=10, color=COL[k], zorder=6, lw=0)        # one dot every 3 latent frames
        ax1.scatter(m[-1, 0], m[-1, 1], s=70 if big else 50, color=COL[k], marker="X", zorder=7, edgecolor="white", lw=0.8)
    m0 = np.mean(np.stack(list(Z["teacher"].values())), 0)[0]
    ax1.scatter(*m0, s=110, facecolor="white", edgecolor="#1f1f1f", lw=1.8, zorder=8)
    ax1.set_xlabel("PC 1 of frame latents (%.0f %% of variance)" % (100 * evr[0])); ax1.set_ylabel("PC 2 (%.0f %% of variance)" % (100 * evr[1]))
    ax1.set_title("Rollout paths: one PCA over all arms, seeds and frames\nthick = mean of seed coordinates (not a real sample)",
                  loc="left", color="#1f1f1f", pad=8, linespacing=1.3)

    # right: distance to the teacher clip with the same seed index over time
    def band(D, col, lw, ls="-", z=3, err=None):
        m = D.mean(0); e = err if err is not None else (D.std(0, ddof=1) / np.sqrt(len(D)) if len(D) > 1 else 0 * m)
        ax2.plot(xs, m, color=col, lw=lw, ls=ls, zorder=z); ax2.fill_between(xs, m - e, m + e, color=col, alpha=0.14, lw=0, zorder=z - 1)
    band(curves["teacher_spread"], REF, 1.8, "--", 4, err=jk_curve)
    for k in order:
        if k in curves and k != "teacher":
            band(curves[k], COL[k], 2.6 if k in OURS else 1.4, ls=LS[k], z=5 if k in OURS else 3)
    for c in range(1, F // 3 + 1): ax2.axvline(c, color="#f0f0f0", lw=0.8, zorder=1)
    ax2.set_xlabel("chunk index (1 chunk = 3 latent frames = 0.75 s)")
    ax2.set_ylabel("relative latent distance |z − z_T| / |z_T|\n(grey dashed = another teacher sample)")
    ax2.set_ylim(bottom=0); ax2.set_xlim(-0.2, xs[-1] + 0.2)
    ax2.set_title("Distance to the teacher clip with the same seed index\n(independent noise except ours 832; mean ± s.e. over seeds)",
                  loc="left", color="#1f1f1f", pad=8, linespacing=1.3)

    # shared legend below both panels
    H = [Line2D([], [], color=COL[k], lw=2.6 if k in KEY else 1.6, ls=LS[k], label=LAB[k]) for k in order]
    H += [Line2D([], [], color=REF, lw=1.8, ls="--", label="teacher vs another teacher seed (right)"),
          Line2D([], [], ls="", marker="o", ms=8, mfc="white", mec="#1f1f1f", mew=1.6, label="start (shared first frame)"),
          Line2D([], [], ls="", marker="X", ms=8, color="#555555", label="last frame (seed mean)")]

    n = {k: len(d) for k, d in L.items()}; nshared = {k: len(curves[k]) for k in curves if k != "teacher_spread"}
    nmin = min(nshared.values())
    ns = sorted(set(n.values()))
    nseed = ("n = %d seeds per arm" % ns[0]) if len(ns) == 1 else ("n = %d seeds per arm but only %d teacher seeds, so every arm is compared on n = %d"
                                                                       % (max(ns), n["teacher"], n["teacher"]))
    upairs = nT * (nT - 1) // 2
    foot = ("Data: %s, %s, first %d latent frames. Sources: means paper/1012/fig_paths_all_%s.json, per-seed values paper/meeting_1012/data/f10_paths_%s.json; "
            "videos %s (latents cached next to each video as reenc_%dx%d.pt); script %s. "
            "Every arm's video is re-encoded with the Wan2.1 VAE on one %d×%d grid (centre-cropped to the teacher's aspect, Waypoint's 2:1 stretch undone). "
            "Left: thin = single seeds, thick = 3-frame moving average. Right: grey band = leave-one-teacher-seed-out jackknife s.e. over %d unique teacher pairs "
            "(s.e. of its time mean = %.0f %%). Ratio = time-mean distance ÷ time-mean teacher-vs-teacher distance. "
            "SDVG draft chunks = mean 'accepted' in sdvg_paths/<scene>/ir_<τ>_s<seed>/loop.json over the seeds used. %s No timing involved."
            % (sc, nseed, F, sc, sc, VID_ROOTS, H2, W2, SCRIPT, H2, W2, upairs, 100 * jk_mean_se / summ["teacher_spread"], CAVEAT))
    where = "%s (n = %d seeds)" % (PRETTY.get(sc, sc), nmin)
    vd = verdict(paired["mean"], paired["se"], "ours 704 closer", "ours 704 farther")
    title = ("%s: ours 704 is %.2f× as far from the teacher as another teacher sample, the 1.3B student %.2f× (paired difference %s ± %.2f, %s)"
             % (where, ratio["ours704"], ratio["student"], sg(paired["mean"]), paired["se"], vd))
    parts = []
    for k in ("sdvg07", "sdvg22", "mg2", "wp", "ours832"):
        if k not in ratio: continue
        t = "%s %.2f×" % (SHORT[k], ratio[k])
        if k in draft_mean: t += " (%.1f of %s chunks from the draft)" % (draft_mean[k], nchunk or "?")
        if k == "ours832": t += " (%s)" % NC
        parts.append(t)
    sub = "Same ratio (1.0 = as far as another teacher sample; > 1 = farther than the teacher's own seed-to-seed spread, ↓ toward 1): " + " · ".join(parts)
    layout(fig, gs, [ax1, ax2], title, sub, foot, dict(handles=H, ncol=3, fontsize=9.2, handlelength=2.4, columnspacing=1.4, labelspacing=0.5))
    for ext in ("png", "svg"): fig.savefig("%s/f10_paths_%s.%s" % (a.out, sc, ext), dpi=200, facecolor="white")
    plt.close(fig)

    out = dict(scene=sc, n=n, n_vs_teacher=nshared, mean_rel_dist=summ, ratio_to_teacher_spread=ratio, ratio_seed_se=ratio_se,
               per_seed_mean_rel_dist={k: {str(s): v for s, v in d.items()} for k, d in per_seed.items()},
               paired_ours704_minus_student=paired, teacher_seeds=tk, teacher_unique_pairs=upairs,
               teacher_spread_jackknife_se=jk_mean_se, teacher_spread_jackknife_rel=jk_mean_se / summ["teacher_spread"],
               sdvg_draft_chunks={k: {str(s): v[0] for s, v in d.items()} for k, d in draft.items()}, sdvg_draft_mean=draft_mean,
               sdvg_chunks_total=nchunk, pca_explained_variance=evr, grid=[H2, W2], latent_frames=F)
    json.dump(out, open("%s/f10_paths_%s.json" % (a.data, sc), "w"), indent=1)
    srcj = "%s/fig_paths_all_%s.json" % (a.src_json, sc)
    if os.path.exists(srcj):
        old = json.load(open(srcj))["mean_rel_dist"]; dmax = max(abs(old[k] - summ[k]) for k in old if k in summ)
        print("[CHECK] %s max |recomputed - %s| = %.2e" % (sc, os.path.basename(srcj), dmax), flush=True)
    print("[FIG]", "%s/f10_paths_%s.png" % (a.out, sc), json.dumps({k: round(v, 3) for k, v in ratio.items()}),
          "paired704-student %+.3f±%.3f n=%d" % (paired["mean"], paired["se"], paired["n"]), flush=True)
    return out


# ---------------------------------------------------------------- summary dot plot (reads meeting_1012/data/f10_paths_<scene>.json)
SC_MARK = {"sand_filling_up_jar": "s", "sandcastle_wave_impact": "D", "easy_lake": "o", "easy_stonehenge_sunset": "^",
           "easy_urban_rain": "v", "pouring_water_into_cup": "P"}


def summary_figure():
    J = {}
    for sc in SCENES:
        p = "%s/f10_paths_%s.json" % (a.data, sc)
        if not os.path.exists(p) or "per_seed_mean_rel_dist" not in json.load(open(p)):
            print("[PENDING] missing per-seed data %s (run the per-scene pass)" % p, flush=True); continue
        J[sc] = json.load(open(p))
        srcj = "%s/fig_paths_all_%s.json" % (a.src_json, sc)                                  # cross-check against the canonical means
        if os.path.exists(srcj):
            old = json.load(open(srcj))["mean_rel_dist"]
            dmax = max(abs(old[k] - J[sc]["mean_rel_dist"][k]) for k in old if k in J[sc]["mean_rel_dist"])
            assert dmax < 1e-4, (sc, dmax); print("[CHECK] summary %s vs %s: max |diff| = %.1e" % (sc, os.path.basename(srcj), dmax), flush=True)
    if len(J) < len(SCENES):
        print("[PENDING] summary not rendered: per-seed data for %d of %d scenes" % (len(J), len(SCENES)), flush=True); return
    scs = list(J); nsc = len(scs)
    arms = [k for _, k, *_ in ARMS if k != "teacher"]
    R = {sc: J[sc]["ratio_to_teacher_spread"] for sc in scs}
    RSE = {sc: J[sc]["ratio_seed_se"] for sc in scs}
    NV = {sc: min(J[sc]["n_vs_teacher"].values()) for sc in scs}
    MEAN = {k: float(np.mean([R[sc][k] for sc in scs])) for k in arms}
    MSE = {k: se([R[sc][k] for sc in scs]) for k in arms}
    RNG = {k: (min(R[sc][k] for sc in scs), max(R[sc][k] for sc in scs)) for k in arms}
    nmax = max(NV.values()); low = [sc for sc in scs if NV[sc] < nmax]

    def across(k1, k2):                                                  # per-scene difference k1 - k2, mean +- s.e. over scenes
        d = [R[sc][k1] - R[sc][k2] for sc in scs]; return float(np.mean(d)), se(d)
    P = {sc: J[sc]["paired_ours704_minus_student"] for sc in scs}
    p_m, p_se = float(np.mean([P[sc]["mean"] for sc in scs])), se([P[sc]["mean"] for sc in scs])
    sig = [sc for sc in scs if abs(P[sc]["mean"]) >= 2 * P[sc]["se"]]

    comp = sorted([k for k in arms if k != "ours832"], key=lambda k: MEAN[k])           # comparable arms, closest to the teacher at the top
    ypos = {k: i for i, k in enumerate(comp)}; ypos["ours832"] = len(comp) + 0.35
    off = dict(zip(scs, np.array([-0.36, -0.24, -0.12, 0.12, 0.24, 0.36])[:nsc] if nsc == 6 else np.linspace(-0.36, 0.36, nsc)))

    fig = plt.figure(figsize=(12, 7.6), dpi=200)
    gs = fig.add_gridspec(1, 1); ax = fig.add_subplot(gs[0]); style(ax); ax.grid(axis="y", visible=False)
    y832 = ypos["ours832"]
    ax.axhspan(y832 - 0.5, y832 + 0.5, facecolor="#f7f5fb", edgecolor="#d9d0ee", hatch="///", lw=0, zorder=0)
    ax.axhline(len(comp) - 0.5 + 0.1, color="#bdbdbd", lw=0.9, zorder=1)
    ax.axvline(1.0, color="#1f1f1f", lw=1.2, ls="--", zorder=2)
    for k in comp + ["ours832"]:
        y0 = ypos[k]; c = COL[k]; nc = k == "ours832"
        for sc in scs:
            hollow = sc in low
            ax.errorbar(R[sc][k], y0 + off[sc], xerr=RSE[sc][k], fmt="none", ecolor=c, elinewidth=0.9, alpha=0.55, capsize=0, zorder=3)
            ax.scatter(R[sc][k], y0 + off[sc], s=22, marker=SC_MARK[sc], facecolor="white" if (hollow or nc) else c, edgecolor=c,
                       lw=1.0, alpha=0.75, zorder=4)
        ax.errorbar(MEAN[k], y0, xerr=MSE[k], fmt="none", ecolor=c, elinewidth=2.6, capsize=4, capthick=2.0, zorder=5)
        ax.scatter(MEAN[k], y0, s=120, marker="o", facecolor="white" if nc else c, edgecolor=c if nc else "white", lw=2.2 if nc else 1.2, zorder=6)
    xmax = max(max(R[sc][k] + RSE[sc][k] for k in arms) for sc in scs)
    ax.set_xlim(0.86, max(1.8, xmax + 0.05)); ax.set_ylim(y832 + 0.62, -0.62)
    tl = []
    for k in comp + ["ours832"]:
        t = SHORT[k]
        if k in SDVG_TAU:
            dm = [J[sc]["sdvg_draft_mean"][k] for sc in scs]; t += "\n(%.1f–%.1f of %d from draft)" % (min(dm), max(dm), J[scs[0]]["sdvg_chunks_total"])
        if k == "ours832": t += "\n(shares the teacher's\nnoise: not comparable)"
        tl.append(t)
    ax.set_yticks([ypos[k] for k in comp + ["ours832"]]); ax.set_yticklabels(tl, fontsize=10, linespacing=1.15)
    for lab, k in zip(ax.get_yticklabels(), comp + ["ours832"]):
        if k == "ours832": lab.set_color("#6b6b6b"); lab.set_fontstyle("italic")
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("time-mean latent distance to the teacher ÷ teacher's own seed-to-seed distance\n"
                  "(1.0 = as far as another teacher sample; > 1 = farther than the teacher's own spread; ↓ toward 1, below 1 is not better)")
    # value column at the right edge of the axes (mean +- s.e. over scenes, range)
    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(1.01, -0.66, "mean ± s.e. over\n%d scenes (range)" % nsc, transform=trans, ha="left", va="bottom", fontsize=9, color=INK2, linespacing=1.15)
    for k in comp + ["ours832"]:
        ax.text(1.01, ypos[k], "%.2f ± %.2f\n(%.2f–%.2f)" % (MEAN[k], MSE[k], *RNG[k]), transform=trans, ha="left", va="center",
                fontsize=9.5, color="#6b6b6b" if k == "ours832" else "#1f1f1f", linespacing=1.1)
    ax.text(1.0, -0.66, "1.0 = as far as another teacher sample ", ha="right", va="bottom", fontsize=9, color="#1f1f1f")
    lo832 = max(R[sc]["ours832"] + RSE[sc]["ours832"] for sc in scs)
    ax.text(lo832 + 0.04, y832, "Even with the teacher's own noise, ours 832 ends as far from the\nteacher as an independent teacher sample (%.2f×)." % MEAN["ours832"],
            ha="left", va="center", fontsize=9, color="#555555", linespacing=1.2, zorder=7,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.9))

    H = [Line2D([], [], ls="", marker=SC_MARK[sc], ms=6.5, mfc="white" if sc in low else "#8a8a8a", mec="#6b6b6b", mew=1.0,
                label="%s%s" % (NICK[sc], (" (n = %d, hollow)" % NV[sc]) if sc in low else "")) for sc in scs]
    H += [Line2D([], [], color="#8a8a8a", lw=0.9, marker="o", ms=4, label="scene value ± s.e. over seeds"),
          Line2D([], [], color="#444444", lw=2.6, marker="o", ms=9, mec="white", label="mean of %d scenes ± s.e. over scenes" % nsc)]

    # SDVG draft counts per scene (seeds used), and the Waypoint quality example from bench-43 (raw: paper/1012/bench43_summary.json)
    dtxt = "; ".join("τ %s %s" % (SDVG_TAU[k].replace("-", "−"), " · ".join(("%s %.1f" % (NICK[sc], J[sc]["sdvg_draft_mean"][k])) if k == "sdvg07"
                     else "%.1f" % J[sc]["sdvg_draft_mean"][k] for sc in scs)) for k in SDVG_TAU) + " (same scene order)"
    b43p = S + "/paper/1012/bench43_summary.json"; ex = ""
    if os.path.exists(b43p) and "sand_filling_up_jar" in R:
        b = json.load(open(b43p))
        ex = (" Example: Waypoint scores %.2f× on jar, closer than a second teacher sample, yet its bench-43 VLM quality is %.2f vs the teacher's %.2f."
              % (R["sand_filling_up_jar"]["wp"], b["wp"]["quality"], b["teacher"]["quality"]))
    jr = {sc: J[sc]["teacher_spread_jackknife_rel"] for sc in scs}; jmax = max(jr, key=jr.get)
    foot = ("Data: paper/meeting_1012/data/f10_paths_<scene>.json (per-seed values from the cached reenc_*.pt latents of the videos under lingbot-world-v2/specwm_out/; "
            "means match paper/1012/fig_paths_all_<scene>.json); SDVG draft counts = 'accepted' in lingbot-world-v2/specwm_out/sdvg_paths/<scene>/ir_<τ>_s<seed>/loop.json; "
            "script %s. %d seeds per arm, but %s have only %d teacher seeds, so all arms there are compared on n = %d (hollow markers). "
            "Distance is to the teacher clip with the same seed index (independent noise except ours 832). Whiskers = s.e. over seeds (arm only); the 1.0 "
            "reference's own jackknife s.e. is %.0f–%.0f %% (largest on %s) and moves all arms of a scene together. "
            "SDVG chunks from the 1.3B draft (of %d, mean over the seeds used): %s. %s%s No fps involved."
            % (SCRIPT, nmax, " and ".join(NICK[sc] for sc in low) or "no scene", min(NV.values()), min(NV.values()),
               100 * min(jr.values()), 100 * max(jr.values()), NICK[jmax], J[scs[0]]["sdvg_chunks_total"], dtxt, CAVEAT, ex))

    mg_o, mg_ose = across("mg2", "ours704"); mg_s, mg_sse = across("mg2", "student"); wp_s, wp_sse = across("wp", "student")
    mg_far = mg_o > 2 * mg_ose and mg_s > 2 * mg_sse
    v704 = verdict(p_m, p_se)
    rel704 = {"within seed noise": "as far from the teacher as", "closer": "closer to the teacher than", "farther": "farther from the teacher than"}[v704]
    title = ("Ours 704 is %s the 1.3B student (%.2f× vs %.2f×%s); Matrix-Game 2.0 is %s (%.2f×); "
             "ours 832 (%.2f×) shares the teacher's noise and is not comparable"
             % (rel704, MEAN["ours704"], MEAN["student"], ", within seed noise" if v704 == "within seed noise" else "",
                "clearly farther" if mg_far else "not clearly farther", MEAN["mg2"], MEAN["ours832"]))
    sig_txt = (" and ".join("%s %s ± %.2f" % (NICK[sc], sg(P[sc]["mean"]), P[sc]["se"]) for sc in sig)) if sig else "no scene"
    sub = ("Paired by seed index, ours 704 − student = %s ± %.2f over %d scenes; only %s reach 2 s.e. "
           "Waypoint %s (%.2f×, range %.2f–%.2f). SDVG τ−0.7 (%.2f×) takes only %.1f–%.1f of %d chunks from the draft, so it is mostly the 14B target."
           % (sg(p_m), p_se, nsc, sig_txt, "varies by scene" if abs(wp_s) < 2 * wp_sse else ("is farther than the student" if wp_s > 0 else "is closer than the student"),
              MEAN["wp"], *RNG["wp"], MEAN["sdvg07"], min(J[sc]["sdvg_draft_mean"]["sdvg07"] for sc in scs),
              max(J[sc]["sdvg_draft_mean"]["sdvg07"] for sc in scs), J[scs[0]]["sdvg_chunks_total"]))
    layout(fig, gs, [ax], title, sub, foot, dict(handles=H, ncol=4, fontsize=9.2, handlelength=1.8, columnspacing=2.0), foot_wrap=185)
    for ext in ("png", "svg"): fig.savefig("%s/f10_paths_summary.%s" % (a.out, ext), dpi=200, facecolor="white")
    plt.close(fig)
    json.dump(dict(ratio=R, ratio_seed_se=RSE, n_vs_teacher=NV, mean_of_scenes=MEAN, se_over_scenes=MSE, range_over_scenes=RNG,
                   paired_ours704_minus_student=dict(per_scene=P, mean=p_m, se_over_scenes=p_se, scenes_beyond_2se=sig),
                   mg2_minus_ours704=[mg_o, mg_ose], mg2_minus_student=[mg_s, mg_sse], wp_minus_student=[wp_s, wp_sse],
                   sdvg_draft_mean={sc: J[sc]["sdvg_draft_mean"] for sc in scs}, teacher_spread_jackknife_rel={sc: J[sc]["teacher_spread_jackknife_rel"] for sc in scs},
                   source=[("%s/f10_paths_%s.json" % (a.data, sc)) for sc in scs]),
              open("%s/f10_paths_summary.json" % a.data, "w"), indent=1)
    print("[FIG]", "%s/f10_paths_summary.png" % a.out, json.dumps({k: [round(MEAN[k], 3), round(MSE[k], 3)] for k in arms}),
          "paired704-student %+.3f±%.3f sig=%s" % (p_m, p_se, sig), flush=True)


if __name__ == "__main__":
    bad = 0
    if not a.summary_only:
        T = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
        for sc in [s for s in a.scenes.split(",") if s]:
            if scene_figure(sc, T) is None: bad += 1
    summary_figure()
    sys.exit(1 if bad else 0)
