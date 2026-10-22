#!/usr/bin/env python
"""f11_modes -- our three operating modes against the 14B teacher on bench-43.

  ours: teacher-parity mode = cw07_safe  (chunk-wise adaptive steps, stop at >= 70 % converged
                                          tokens, safe commit)
  ours: quality mode        = ours832    (832, 2-step, fused + partial commit)
  ours: realtime mode       = ours704    (704, 2-step, fused + partial commit)

Panel a: VLM quality (mean +/- 1 s.e. over 86 clips) vs fps (under load) for the teacher, the four
         chunk-wise arms (cw07_sep, cw07_pc1, cw07_safe, cw05_pc1), ours832 and ours704; the four
         chunk-wise arms sit within 1.1 fps of each other, so an inset zooms on them.
Panel b: paired delta quality vs the teacher (mean +/- 1.96 s.e., 86 pairs) for the same arms, with a
         number table (delta quality, delta drift, mean denoising steps, fps and speed-up).

Data:
- paper/1012/bench43_summary.json (scripts/bench43_report.py --json): fps, quality, quality_se,
  drift, drift_se, d_quality, d_quality_se, d_drift, d_drift_se, paired_n, clips.
- mean denoising steps: per-clip loop.json "mean_steps" under
  lingbot-world-v2/specwm_out/bench43/<task>/<dir>_s{1234,4321}/loop.json, averaged over the 86 clips
  (teacher = uniform4, cw arms = their own names). ours832 / ours704 run a fixed 2-step schedule and
  their loop.json has no mean_steps field -> 2.00 (facts.md section 12). The values are checked
  against facts.md section 12 (4.00 / 2.91 / 3.01 / 3.01 / 2.61 / 2.00); the facts values are used if
  the logs are missing.
- quiet timing pass: facts.md section 15 table (teacher 4.08 fps quiet steady; speed-ups of the three modes
  1.67x / 2.39x / 3.85x), read at run time. The title and the table's speed column use these quiet steady
  values (facts.md section 15: "prefer the quiet steady ratios in slides"); the title adds the bench-43
  under-load ratio in parentheses, and panel a keeps the bench-43 under-load fps axis. The teacher's older
  "4.9 fps quiet" is a whole-clip average (facts.md section 15 correction) and is no longer used.

Layout checks printed on every run: text vs text, text and inset frame vs plot elements (error-bar
stems/caps, markers, reference lines, zoom box), inset labels inside the inset, leader clearance.

Output: paper/meeting_1012/figs/f11_modes.{png,svg}
Run:    CUDA_VISIBLE_DEVICES="" /home/hk28965/specwm/venv/bin/python \
        /home/hk28965/specwm/paper/meeting_1012/scripts/f11_modes.py
"""
import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.transforms import Bbox, blended_transform_factory  # noqa: E402

ROOT = "/home/hk28965/specwm"
SRC = ROOT + "/paper/1012/bench43_summary.json"
BENCH = ROOT + "/lingbot-world-v2/specwm_out/bench43"
OUT = ROOT + "/paper/meeting_1012/figs"
FID = "f11_modes"
Z = 1.96
FACTS = ROOT + "/paper/meeting_1012/facts.md"


def quiet_steady():
    """facts.md section 15 (quiet timing pass, steady s/chunk, jar + castle): (fps, speed-up) as written there"""
    sec = open(FACTS, encoding="utf-8").read().split("\n## 15.")[1]
    rows = {}
    for line in sec.splitlines():
        if line.startswith("|"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            rows[c[0]] = c
    key = {"teacher": "teacher 14B 4-step (uniform4)",
           "cw07_sep": "chunk-wise 0.7, separate commit",
           "cw07_pc1": "chunk-wise 0.7, fused + partial",
           "cw07_safe": "ours teacher-parity mode (chunk-wise 0.7, safe commit)",
           "cw05_pc1": "chunk-wise 0.5, fused + partial",
           "ours832": "ours quality mode (832 fused + partial)",
           "ours704": "ours realtime mode (704 fused + partial)"}
    return {a: (rows[k][3], rows[k][4]) for a, k in key.items()}


QS = quiet_steady()
TEACHER_QUIET_FPS = QS["teacher"][0]  # 4.08 = steady quiet (facts.md section 15), NOT the whole-clip 4.9

# rows top -> bottom = ascending fps
ARMS = ["teacher", "cw07_sep", "cw07_safe", "cw07_pc1", "cw05_pc1", "ours832", "ours704"]
MODES = {"cw07_safe": "ours: teacher-parity mode", "ours832": "ours: quality mode",
         "ours704": "ours: realtime mode"}
NAME = {"teacher": "teacher 14B 4-step", "cw07_sep": "cw07_sep", "cw07_pc1": "cw07_pc1",
        "cw05_pc1": "cw05_pc1", **MODES}
DESC = {"teacher": "reference (Δ = 0)",
        "cw07_sep": "chunk-wise 0.7, separate commit",
        "cw07_safe": "chunk-wise 0.7, safe commit",
        "cw07_pc1": "chunk-wise 0.7, fused + partial",
        "cw05_pc1": "chunk-wise 0.5, fused + partial",
        "ours832": "≈832×480, 2-step, fused + partial",
        "ours704": "≈704×384, 2-step, fused + partial"}

# fixed meeting colours; teacher-parity mode = new blue, the other chunk-wise arms lighter blues
COLOR = {"teacher": "#1f1f1f", "cw07_safe": "#1565c0", "cw07_sep": "#4a8bd6",
         "cw07_pc1": "#74a9e4", "cw05_pc1": "#9cc2ec", "ours832": "#7b4bd1", "ours704": "#d81b60"}
MARKER = {"teacher": "s", "cw07_safe": "*", "ours832": "*", "ours704": "*",
          "cw07_sep": "o", "cw07_pc1": "D", "cw05_pc1": "v"}
MS = {"teacher": 10, "cw07_safe": 17, "ours832": 17, "ours704": 17,
      "cw07_sep": 8, "cw07_pc1": 7, "cw05_pc1": 8.5}
ZO = {"teacher": 4.6, "cw07_sep": 4, "cw07_pc1": 4, "cw05_pc1": 4,
      "cw07_safe": 5, "ours832": 5, "ours704": 5}

# loop.json directory per arm (same map as scripts/bench43_report.py)
LOOP = {"teacher": "uniform4", "cw07_sep": "cw07_sep", "cw07_safe": "cw07_safe",
        "cw07_pc1": "cw07_pc1", "cw05_pc1": "cw05_pc1"}
FACT_STEPS = {"teacher": 4.00, "cw07_sep": 2.91, "cw07_safe": 3.01, "cw07_pc1": 3.01,
              "cw05_pc1": 2.61, "ours832": 2.00, "ours704": 2.00}  # facts.md section 12

INK, INK2, MUTED, FOOT, GRID = "#1f1f1f", "#4a4a4a", "#6b6b6b", "#7f7f7f", "#e6e6e6"

FIG_W, FIG_H = 14.0, 8.0

# panel a layout: inset x0, y0, w in main-axes fraction (its height is fitted under the zoom box) and
# its data x range (fps); teacher-parity leader: points left of the star centre where it lands,
# label bottom above the zoom box (points), label start left of the leader (fps)
INSET_X0, INSET_Y0, INSET_W = 0.075, 0.09, 0.45
INSET_X = (5.60, 7.55)
LEAD_END = 3.8    # leader lands this many points left of the star centre (on the arm's top edge)
LEAD_SLANT = 8.0  # leader top is this many points further left (clears the star's own upper cap)
LABEL_UP = 6.0
LABEL_LEFT = 0.10

plt.rcParams.update({
    "font.family": "DejaVu Sans", "svg.fonttype": "none",
    "axes.titlesize": 11, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.edgecolor": "#9a9a9a", "axes.linewidth": 0.8,
    "xtick.color": "#555555", "ytick.color": "#555555",
})


def sgn(v, nd):
    return ("+" if v >= 0 else "−") + f"{abs(v):.{nd}f}"


def mean_steps(arm):
    """mean over clips of loop.json mean_steps (None if the logs are missing)"""
    if arm not in LOOP:
        return None
    v = []
    for f in sorted(glob.glob(f"{BENCH}/*/{LOOP[arm]}_s*/loop.json")):
        if f.endswith("_warm/loop.json"):
            continue
        r = json.load(open(f))
        if r.get("mean_steps") is not None:
            v.append(float(r["mean_steps"]))
    return (float(np.mean(v)), len(v)) if v else None


def speedup(arm, S):
    """fps ratio to the teacher (both under load); 2 decimals for the chunk-wise arms as in facts.md
    section 12 (1.51x ...), 1 decimal for ours 832 / 704 (2.5x, 4.1x)"""
    r = S[arm]["fps"] / S["teacher"]["fps"]
    return f"{r:.1f}×" if arm in ("ours832", "ours704") else f"{r:.2f}×"


def wrap_measured(fig, sentences, fontsize, max_px):
    r = fig.canvas.get_renderer()
    probe = fig.text(0, 0, "", fontsize=fontsize)
    lines, cur = [], ""
    for word in "  ".join(sentences).split(" "):
        trial = word if not cur else cur + " " + word
        probe.set_text(trial)
        if probe.get_window_extent(r).width > max_px and cur:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    lines.append(cur)
    probe.remove()
    return [ln.strip() for ln in lines]


def check_overlaps(fig, ignore=()):
    """print any pair of visible texts whose boxes intersect, and any text outside the canvas"""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    ts = [t for t in fig.findobj(matplotlib.text.Text)
          if t.get_visible() and t.get_text().strip() and t not in ignore]
    bb = [(t, t.get_window_extent(r)) for t in ts]
    bad = 0
    for i, (t, b) in enumerate(bb):
        if b.x0 < 0 or b.y0 < 0 or b.x1 > W or b.y1 > H:
            print("OUTSIDE:", repr(t.get_text()[:50]))
            bad += 1
        for t2, b2 in bb[i + 1:]:
            if b.overlaps(b2):
                print("OVERLAP:", repr(t.get_text()[:40]), "<>", repr(t2.get_text()[:40]))
                bad += 1
    print("text overlap check:", "OK" if bad == 0 else f"{bad} problems")
    return bad


def _artist_shapes(ax):
    """display-space shapes of the data artists of one axes: (name, path) for line/edge paths and
    (name, Bbox) for marker boxes (half-size = markersize / 2)"""
    px = ax.figure.dpi / 72.0
    paths, boxes = [], []
    for c in ax.collections:  # errorbar stems
        if not c.get_visible():
            continue
        tr = c.get_transform()
        for p in c.get_paths():
            paths.append((f"{c.get_label() or 'collection'}", p.transformed(tr)))
    for ln in ax.lines:  # reference lines, errorbar caps, markers
        if not ln.get_visible():
            continue
        tp = ln.get_transform().transform_path(ln.get_path())
        if ln.get_linestyle() not in ("None", "", " ", "none") and len(tp.vertices) > 1:
            paths.append((ln.get_label() or "line", tp))
        if ln.get_marker() not in (None, "None", "", " ", "none"):
            h = 0.5 * ln.get_markersize() * px
            t = 0.5 * max(ln.get_markeredgewidth(), 1.0) * px
            # error-bar caps are thin bars, everything else a square of the marker size
            hx, hy = {"_": (h, t), "|": (t, h)}.get(ln.get_marker(), (h, h))
            for x, y in tp.vertices:
                boxes.append((f"marker {ln.get_marker()} of {ln.get_color()}",
                              Bbox([[x - hx, y - hy], [x + hx, y + hy]])))
    for p in ax.patches:  # zoom rectangle edges
        paths.append(("patch edge", p.get_transform().transform_path(p.get_path())))
    return paths, boxes


def check_artist_overlaps(fig, axes, texts, frames=(), pad_pt=1.5):
    """text vs plot elements (error bars, caps, markers, reference lines, patch edges) of `axes`, and
    `frames` = (name, axes) whose frame + tick labels must not touch the data artists of `axes`
    (e.g. an inset over the main panel). The text-only check above cannot see these collisions."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pad = pad_pt * fig.dpi / 72.0
    shapes = [(a, *_artist_shapes(a)) for a in axes]
    bad = 0

    def hits(name, box, skip_ax=None):
        n = 0
        for a, paths, boxes in shapes:
            if a is skip_ax:
                continue
            for nm, p in paths:
                if p.intersects_bbox(box, filled=False):
                    print(f"ARTIST OVERLAP: {name} <> {nm}")
                    n += 1
            for nm, b in boxes:
                if box.overlaps(b):
                    print(f"ARTIST OVERLAP: {name} <> {nm}")
                    n += 1
        return n

    for t in texts:
        if not (t.get_visible() and t.get_text().strip()):
            continue
        b = t.get_window_extent(r).padded(pad)
        bad += hits(repr(t.get_text()[:40]), b)
    for name, fa in frames:  # the frame itself; its tick labels and title are among `texts`
        bad += hits(name, fa.get_window_extent(r).padded(pad), skip_ax=fa)
    print("text/frame vs plot-element check:", "OK" if bad == 0 else f"{bad} problems")
    return bad


def main():
    S = json.load(open(SRC))
    for k in ARMS:
        for f in ("fps", "quality", "quality_se"):
            assert S[k].get(f) is not None, (k, f)
        if k != "teacher":
            for f in ("d_quality", "d_quality_se", "d_drift", "d_drift_se", "paired_n"):
                assert S[k].get(f) is not None, (k, f)
    T = S["teacher"]
    npairs = sorted({S[k]["paired_n"] for k in ARMS if k != "teacher"})
    nclips = sorted({S[k].get("clips", 86) for k in ARMS})

    steps, steps_src = {}, {}
    for k in ARMS:
        ms = mean_steps(k)
        if ms is None:
            steps[k], steps_src[k] = FACT_STEPS[k], "facts §12"
        else:
            steps[k], steps_src[k] = ms[0], f"loop.json mean over {ms[1]} clips"
            if f"{ms[0]:.2f}" != f"{FACT_STEPS[k]:.2f}":
                print(f"[warn] {k}: loop.json mean steps {ms[0]:.4f} != facts {FACT_STEPS[k]:.2f}")
    print("mean steps:", {k: (round(steps[k], 4), steps_src[k]) for k in ARMS})

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    dpi = fig.dpi

    # ------------------------------------------------------------------ footer (measured wrap)
    sentences = [
        f"Data: bench-43 = 38 StEvo camera-WM tasks + 5 easy scenes × seeds 1234/4321 (161 frames) = "
        f"{'/'.join(map(str, nclips))} clips per arm; Δ = arm − teacher on the same prompt and seed "
        f"({'/'.join(map(str, npairs))} pairs). Judge: Qwen3-VL-32B-Instruct, 8 frames at 448 px; drift "
        "prompt AUROC 0.82 vs human labels, quality prompt not validated against humans.",
        "a: error bars ±1 s.e. over clips. b: marker = mean paired Δ, bar = ±1.96 s.e.; table = mean ± s.e. "
        "(bold = best, underline = second, by mean).",
        "fps = 12 / steady s per chunk on one RTX PRO 6000 Blackwell (not H100). Panel a: bench-43, measured under "
        f"load (7 GPU jobs at once; teacher {T['fps']:.1f} fps). Title speed-ups and table fps: quiet timing pass "
        "(facts.md §15: jar + sandcastle × 3 seeds, generation only, no other generation jobs; teacher "
        f"{TEACHER_QUIET_FPS} fps); the bench-43 under-load ratios ({speedup('cw07_safe', S)} / "
        f"{speedup('ours832', S)} / {speedup('ours704', S)}) agree within ~5 %. The teacher's older "
        "“4.9 fps quiet” was a whole-clip average. Final VAE decode excluded (ours decodes on the device).",
        "Chunk-wise arms: 14B 4-step schedule; after each step a learned head (head14b_v3) predicts "
        "per-token convergence and the chunk stops once ≥ 70 % (cw07) / ≥ 50 % (cw05) of tokens are "
        "predicted converged. Commit: separate = clean full pass; partial = fused + clean pass on the "
        "newest latent frame; safe = partial if the chunk stopped at step ≥ 3, else a full clean pass.",
        "Mean steps = loop.json mean_steps averaged over clips; ours 832 / 704 = fixed 2-step schedule. "
        "Source: paper/1012/bench43_summary.json, lingbot-world-v2/specwm_out/bench43/*/<arm>_s*/loop.json; "
        "facts.md §2, §12, §15.",
    ]
    foot_lines = wrap_measured(fig, sentences, 9, 0.965 * FIG_W * dpi)
    lh = 9 * 1.38 / 72.0
    foot_h = len(foot_lines) * lh
    ax_bot_in = 0.08 + foot_h + 0.95
    ax_top_in = FIG_H - 1.62
    yb, hh = ax_bot_in / FIG_H, (ax_top_in - ax_bot_in) / FIG_H

    axA = fig.add_axes([0.052, yb, 0.323, hh])
    axB = fig.add_axes([0.597, yb, 0.148, hh])

    # ------------------------------------------------------------------ panel a: quality vs fps
    ax = axA
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    xmax = 18.0
    # ylo below the lowest data point (6.82) leaves room for the inset under the zoom box; yhi leaves
    # room for the teacher-parity label above the zoom box
    ylo, yhi = 6.40, 8.00
    ax.set_xlim(0, xmax)
    ax.set_ylim(ylo, yhi)
    ax.set_yticks(np.arange(6.4, 8.01, 0.2))
    ax.set_xlabel("Throughput, fps (under load)   ↑ higher is better")
    ax.set_ylabel("VLM visual quality (1–10)   ↑ higher is better")
    ax.set_title("a   Visual quality vs speed", loc="left", fontsize=11, color=INK, pad=8)

    # teacher reference line (from the teacher marker to the right); its label sits below the line
    ax.plot([T["fps"], xmax], [T["quality"]] * 2, color="#9e9e9e", ls=(0, (5, 3)), lw=1.1, zorder=1)
    ax.text(xmax - 0.15, T["quality"] - 0.012, f"teacher quality {T['quality']:.2f}", ha="right",
            va="top", fontsize=9, color=MUTED, zorder=3)

    def draw_pt(a, k, scale=1.0):
        r = S[k]
        hollow = k == "teacher"
        a.errorbar(r["fps"], r["quality"], yerr=r["quality_se"], fmt=MARKER[k], ms=MS[k] * scale,
                   color=COLOR[k], mfc="none" if hollow else COLOR[k],
                   mec=COLOR[k] if hollow else "white", mew=1.8 if hollow else 1.0,
                   ecolor=COLOR[k], elinewidth=1.4, capsize=3, capthick=1.4, zorder=ZO[k])

    for k in ARMS:
        draw_pt(ax, k)

    # zoom box around the chunk-wise cluster (data range of the inset)
    cw = ["cw07_sep", "cw07_safe", "cw07_pc1", "cw05_pc1"]
    zx0, zx1 = INSET_X
    zy0 = min(S[k]["quality"] - S[k]["quality_se"] for k in cw) - 0.10
    zy1 = max(S[k]["quality"] + S[k]["quality_se"] for k in cw) + 0.05
    ax.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, facecolor="none", edgecolor="#9a9a9a",
                           lw=0.8, zorder=3))

    def block(k, xy, dx, dy, ha):
        """three-line mode label, bottom line at offset (dx, dy) points from xy:
        bold name / fps (speed-up vs teacher) / quality"""
        r = S[k]
        lines = [(f"quality {r['quality']:.2f}", 0, "normal", 9.5),
                 (f"{QS[k][1]} quiet ({speedup(k, S)} under load)", 12, "normal", 9.5),
                 (MODES[k], 25, "bold", 10.5)]
        for txt, up, fw, fs in lines:
            ax.annotate(txt, xy, xytext=(dx, dy + up), textcoords="offset points", ha=ha,
                        va="bottom", fontsize=fs, color=COLOR[k], fontweight=fw, zorder=6)

    # teacher: left of its marker (its numbers are in the table of panel b and on the line label)
    ax.annotate("teacher\n14B 4-step", (T["fps"], T["quality"]), xytext=(-11, 0),
                textcoords="offset points", ha="right", va="center", color=INK, fontsize=10,
                linespacing=1.2, zorder=6)
    # teacher-parity: label above the zoom box; a near-vertical leader drops from it onto the flat top
    # edge of the star's upper-left arm, through the gap between cw07_sep's error bar and the star's
    # own upper cap. (A leader from the right, as before, ended on cw07_pc1's error bar 0.1 fps right
    # of the star, whose diamond is hidden under the star; the inset separates those arms.) The leader
    # runs under the star (zorder 2 < 5), so it visibly ends inside the star's arm.
    p = S["cw07_safe"]
    ppx = fig.dpi / 72.0
    cx, cy = ax.transData.transform((p["fps"], p["quality"]))
    # '*' marker of size MS: upper-left arm's flat top edge at y = +0.155 MS, from the tip
    # (x = -0.476 MS) to the inner vertex (x = -0.112 MS); the white 1 pt edge trims the tip
    end_px = (cx - LEAD_END * ppx, cy + 0.085 * MS["cw07_safe"] * ppx)
    y_lab = ax.transData.transform((0, zy1))[1] + LABEL_UP * ppx  # bottom of the label block
    top_px = (end_px[0] - LEAD_SLANT * ppx, y_lab - 2.5 * ppx)
    inv_d = ax.transData.inverted()
    (xe, ye), (xt, yt) = inv_d.transform([end_px, top_px])
    lead = ax.plot([xe, xt], [ye, yt], color="#9a9a9a", lw=0.8, zorder=2, solid_capstyle="butt",
                   label="teacher-parity leader")[0]
    block("cw07_safe", (xt - LABEL_LEFT, zy1), 0, LABEL_UP, "left")
    # quality mode right of its star; realtime mode below its star (above it the two blocks collide)
    block("ours832", (S["ours832"]["fps"], S["ours832"]["quality"]), 13, 3, "left")
    q7r = S["ours704"]
    block("ours704", (q7r["fps"], q7r["quality"] - q7r["quality_se"]), 30, -42, "right")

    # inset: the four chunk-wise arms
    # inset raised clear of the main x axis (its tick labels carry 'fps' so they cannot be read as main
    # ticks), and it ends left of ours832 (9.8 fps) so neither the frame nor the title meets its error bar
    # height: the inset title (pad 3 pt + 9 pt text) plus a 5 pt gap must stay below the zoom box
    ax_h_pt = ax.get_position().height * FIG_H * 72.0
    zb_frac = (zy0 - ylo) / (yhi - ylo)
    inset_h = zb_frac - INSET_Y0 - (3 + 9 * 1.25 + 5) / ax_h_pt
    axI = ax.inset_axes([INSET_X0, INSET_Y0, INSET_W, inset_h])
    axI.set_xlim(zx0, zx1)
    axI.set_ylim(zy0, zy1)
    axI.set_facecolor("white")
    for sp in axI.spines.values():
        sp.set_edgecolor("#9a9a9a")
        sp.set_linewidth(0.8)
    axI.grid(True, color="#efefef", lw=0.7, zorder=0)
    axI.set_axisbelow(True)
    axI.tick_params(labelsize=8, length=2, pad=1.5, colors="#666666")
    axI.set_xticks([6.0, 6.5, 7.0])
    axI.set_xticklabels(["6.0 fps", "6.5 fps", "7.0 fps"])
    axI.set_yticks([7.2, 7.3, 7.4, 7.5, 7.6])
    axI.axhline(T["quality"], color="#9e9e9e", ls=(0, (5, 3)), lw=1.0, zorder=1)
    for k in cw:
        draw_pt(axI, k, scale=0.65 if k == "cw07_safe" else 0.9)
    il = dict(fontsize=9, zorder=6, linespacing=1.15)

    def ilab(k, txt, off, ha, va, bold=False, leader=False):
        r = S[k]
        axI.annotate(txt, (r["fps"], r["quality"]), xytext=off, textcoords="offset points", ha=ha,
                     va=va, color=COLOR[k] if bold else INK, fontweight="bold" if bold else "normal",
                     **il)
        if leader:
            axI.annotate("", (r["fps"], r["quality"]), xytext=off, textcoords="offset points",
                         arrowprops=dict(arrowstyle="-", color="#9a9a9a", lw=0.7, shrinkA=0,
                                         shrinkB=MS[k] * 0.45), zorder=2)

    ilab("cw07_sep", "cw07_sep", (7, 2), "left", "center")
    ilab("cw07_safe", "teacher-parity\n(cw07_safe)", (-8, -5), "right", "top", bold=True)
    # pc1 / cw05 sit 0.1 fps from their neighbours: label directly above / below their own error bars
    for k, dy, va in (("cw07_pc1", 0.035, "bottom"), ("cw05_pc1", -0.015, "top")):
        r = S[k]
        ye = r["quality"] + (r["quality_se"] if dy > 0 else -r["quality_se"]) + dy
        if k == "cw07_pc1":  # clear of the teacher line and of the safe arm's upper error bar
            ye = max(ye, T["quality"] + 0.02, S["cw07_safe"]["quality"] + S["cw07_safe"]["quality_se"] + 0.015)
        # cw05_pc1 centred under its bar (left-aligned it ran past the inset's right edge)
        axI.text(r["fps"] - 0.03 if dy > 0 else r["fps"], ye, k, ha="left" if dy > 0 else "center",
                 va=va, fontsize=9, color=INK, zorder=6)
    axI.set_title("zoom: chunk-wise arms", fontsize=9, color=MUTED, pad=3, loc="left")

    # ------------------------------------------------------------------ panel b: paired Δ quality
    n = len(ARMS)
    ypos = {k: n - 1 - i for i, k in enumerate(ARMS)}
    axB.patch.set_alpha(0)  # let the highlight bands show through
    for s in ("top", "right", "left"):
        axB.spines[s].set_visible(False)
    axB.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    axB.tick_params(axis="y", length=0)
    axB.set_yticks([])
    ytop = n - 0.4 + 0.25
    axB.set_ylim(-0.6, n - 0.4 + 0.45)
    for k in ARMS:
        axB.axhline(ypos[k] - 0.5, color="#f0f0f0", lw=0.8, zorder=0)
    axB.axvline(0, color=COLOR["teacher"], lw=1.6, zorder=2)
    for k in ARMS:
        y = ypos[k]
        if k == "teacher":
            axB.plot([0], [y], marker="s", ms=10, mfc="white", mec=COLOR[k], mew=1.8, zorder=4)
            continue
        m, se = S[k]["d_quality"], S[k]["d_quality_se"]
        lo, hi = m - Z * se, m + Z * se
        lw = 2.4 if k in MODES else 1.8
        axB.plot([lo, hi], [y, y], color=COLOR[k], lw=lw, solid_capstyle="round", zorder=3)
        for e in (lo, hi):
            axB.plot([e, e], [y - 0.13, y + 0.13], color=COLOR[k], lw=1.5, zorder=3)
        axB.plot([m], [y], marker=MARKER[k], ms=MS[k] * (0.85 if k in MODES else 1.0),
                 mfc=COLOR[k], mec="white", mew=1.2, zorder=4)
    dlo = min(S[k]["d_quality"] - Z * S[k]["d_quality_se"] for k in ARMS if k != "teacher")
    dhi = max(S[k]["d_quality"] + Z * S[k]["d_quality_se"] for k in ARMS if k != "teacher")
    axB.set_xlim(dlo - 0.10, max(dhi, 0) + 0.10)
    axB.set_xticks([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2])
    axB.set_xticklabels(["−0.8", "−0.6", "−0.4", "−0.2", "0", "+0.2"], fontsize=9)
    axB.set_xlabel("Δ VLM quality vs teacher\n(1–10 scale)   ↑ higher is better", fontsize=11,
                   color=INK)
    axB.text(0, ytop, "teacher = 0  ", ha="right", va="center", fontsize=9, color=INK2)
    figx_title = 0.405
    fig.text(figx_title, axB.get_position().y1 + 8 / 72 / FIG_H, "b   Paired Δ quality vs teacher "
             f"(mean ± 1.96 s.e., {'/'.join(map(str, npairs))} pairs) and cost", ha="left",
             va="bottom", fontsize=11, color=INK)

    # row labels (left of panel b) and the number table (right of panel b)
    tr = blended_transform_factory(fig.transFigure, axB.transData)
    x_lab = 0.585
    # highlight bands for the three modes
    for k in MODES:
        fig.patches.append(Rectangle((figx_title, ypos[k] - 0.47), 0.995 - figx_title, 0.94,
                                     transform=tr, facecolor="#f5f7fa", edgecolor="none",
                                     zorder=-1))
    for k in ARMS:
        y = ypos[k]
        bold = k in MODES
        fig.text(x_lab, y + 0.16, NAME[k], transform=tr, ha="right", va="center", fontsize=10.5,
                 color=COLOR[k] if bold else INK, fontweight="bold" if bold else "normal")
        fig.text(x_lab, y - 0.20, DESC[k], transform=tr, ha="right", va="center", fontsize=9,
                 color=MUTED)

    cols = [  # (x centre in fig coords, header, per-arm text, key for best/second, lower is better)
        (0.788, "Δ quality ↑\nmean ± s.e.",
         lambda k: "–" if k == "teacher" else f"{sgn(S[k]['d_quality'], 2)} ± {S[k]['d_quality_se']:.2f}",
         lambda k: None if k == "teacher" else S[k]["d_quality"], False),
        (0.858, "Δ drift ↓\nmean ± s.e.",
         lambda k: "–" if k == "teacher" else f"{sgn(S[k]['d_drift'], 1)} ± {S[k]['d_drift_se']:.1f}",
         lambda k: None if k == "teacher" else S[k]["d_drift"], True),
        (0.912, "mean\nsteps ↓",
         lambda k: f"{steps[k]:.2f}", lambda k: round(steps[k], 2), True),
        (0.964, "quiet fps ↑\n(×teacher)",
         lambda k: f"{QS[k][0]}\n({QS[k][1]})",
         lambda k: float(QS[k][0]), False),
    ]
    under = []
    for xc, head, fmt, key, low in cols:
        fig.text(xc, ytop + 0.05, head, transform=tr, ha="center", va="center", fontsize=9,
                 color=INK2, linespacing=1.15)
        vals = {k: key(k) for k in ARMS if key(k) is not None}
        uniq = sorted(set(vals.values()), reverse=not low)
        best = uniq[0]
        second = uniq[1] if len(uniq) > 1 else None
        for k in ARMS:
            txt = fmt(k)
            v = vals.get(k)
            t = fig.text(xc, ypos[k], txt, transform=tr, ha="center", va="center",
                         fontsize=9.5 if "\n" not in txt else 9, color=INK, linespacing=1.1,
                         fontweight="bold" if v is not None and v == best else "normal")
            if v is not None and v == second:
                under.append(t)

    # ------------------------------------------------------------------ title, subtitle, footer
    sp, q8, q7 = S["cw07_safe"], S["ours832"], S["ours704"]
    # lead with the quiet steady speed-ups (facts.md section 15, preferred for slides); the bench-43 under-load
    # ratio (facts.md section 12: 1.74x, 2.5x, 4.1x) in parentheses
    title = (f"Teacher-level VLM quality at {QS['cw07_safe'][1]} quiet ({speedup('cw07_safe', S)} under load); "
             f"Δ quality {sgn(q8['d_quality'], 2)} at {QS['ours832'][1]} ({speedup('ours832', S)}), "
             f"{sgn(q7['d_quality'], 2)} at {QS['ours704'][1]} ({speedup('ours704', S)})")
    tie_q = abs(sp["d_quality"]) < Z * sp["d_quality_se"]
    tie_d = abs(sp["d_drift"]) < Z * sp["d_drift_se"]
    sub = [
        "Modes: teacher-parity = chunk-wise adaptive steps + safe commit (cw07_safe) · quality = 832, "
        "2 steps, fused + partial commit · realtime = the same at 704.",
        f"Teacher-parity: Δ quality {sgn(sp['d_quality'], 2)} ± {sp['d_quality_se']:.2f}, Δ drift "
        f"{sgn(sp['d_drift'], 1)} ± {sp['d_drift_se']:.1f} vs teacher "
        f"({'both 95 % CIs include 0' if tie_q and tie_d else 'paired'}). Speed-ups (teacher-parity / quality / "
        f"realtime): {QS['cw07_safe'][1]} / {QS['ours832'][1]} / {QS['ours704'][1]} quiet steady (facts §15, teacher "
        f"{TEACHER_QUIET_FPS} fps; table); {speedup('cw07_safe', S)} / {speedup('ours832', S)} / "
        f"{speedup('ours704', S)} on bench-43 under load (panel a, teacher {T['fps']:.1f} fps).",
    ]
    fig.text(0.012, 1 - 0.20 / FIG_H, title, ha="left", va="top", fontsize=13, color=INK,
             fontweight="bold")
    sub = wrap_measured(fig, sub[:1], 10.5, 0.975 * FIG_W * dpi) + \
        wrap_measured(fig, sub[1:], 10.5, 0.975 * FIG_W * dpi)
    fig.text(0.012, 1 - 0.62 / FIG_H, "\n".join(sub), ha="left", va="top", fontsize=10.5,
             color=INK2, linespacing=1.45)
    fig.text(0.012, 0.08 / FIG_H, "\n".join(foot_lines), ha="left", va="bottom", fontsize=9,
             color=FOOT, linespacing=1.38)

    # underline the second-best table cells (after layout is final)
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    for t in under:
        b = t.get_window_extent(rr)
        lines = t.get_text().split("\n")
        # for two-line cells underline the first line only
        y_line = b.y0 - 1.5 if len(lines) == 1 else b.y0 + b.height / 2 + 0.5
        w_first = b.width
        if len(lines) > 1:
            probe = fig.text(0, 0, lines[0], fontsize=t.get_fontsize())
            w_first = probe.get_window_extent(rr).width
            probe.remove()
        xc_px = 0.5 * (b.x0 + b.x1)
        (x0, y0) = inv.transform((xc_px - w_first / 2, y_line))
        (x1, _) = inv.transform((xc_px + w_first / 2, y_line))
        fig.lines.append(Line2D([x0, x1], [y0, y0], transform=fig.transFigure, color=INK, lw=0.9))

    # inset/axes vs text sanity
    bad = check_overlaps(fig)
    # texts and the inset frame vs error bars / caps / markers / reference lines / zoom box
    texts_a = [t for t in fig.findobj(matplotlib.text.Text)
               if t.get_visible() and t.get_text().strip()]
    bad += check_artist_overlaps(fig, [axA, axI], texts_a, frames=[("inset frame", axI)])
    # inset labels must stay inside the inset frame
    fb = axI.get_window_extent(rr)
    for t in axI.texts:
        b = t.get_window_extent(rr)
        if b.x0 < fb.x0 or b.x1 > fb.x1 or b.y0 < fb.y0 or b.y1 > fb.y1:
            print("OUTSIDE INSET:", repr(t.get_text()))
            bad += 1
    # leader clearance: closest error-bar cap / marker other than the star it points at (points)
    _, mboxes = _artist_shapes(axA)
    lp = lead.get_transform().transform(np.c_[lead.get_xdata(), lead.get_ydata()])
    pts = lp[0] + np.linspace(0, 1, 400)[:, None] * (lp[1] - lp[0])
    star_c = np.array([cx, cy])
    gaps = []
    for nm, b in mboxes:
        if nm.startswith("marker *") and np.allclose([b.x0 + b.width / 2, b.y0 + b.height / 2], star_c):
            continue
        dx = np.maximum(np.maximum(b.x0 - pts[:, 0], pts[:, 0] - b.x1), 0)
        dy = np.maximum(np.maximum(b.y0 - pts[:, 1], pts[:, 1] - b.y1), 0)
        gaps.append((float(np.hypot(dx, dy).min()) / ppx, nm))
    g = sorted(gaps)[:3]
    print("leader clearance (pt):", [(round(d, 1), nm) for d, nm in g])
    if g[0][0] < 2.0:
        print("[warn] leader passes within 2 pt of", g[0][1])
        bad += 1
    tw = fig.texts[[t.get_text() for t in fig.texts].index(title)].get_window_extent(rr).width
    print(f"title width {tw:.0f} px of {fig.bbox.width:.0f}")
    W, H = fig.bbox.width, fig.bbox.height
    for a in (axA, axB):
        b = a.get_tightbbox(rr)
        if b.x0 < 0 or b.x1 > W:
            print("[warn] axes runs off the canvas", b)

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{FID}.png", dpi=200, facecolor="white")
    fig.savefig(f"{OUT}/{FID}.svg", facecolor="white")
    print(f"{OUT}/{FID}.png")
    print(f"{OUT}/{FID}.svg")
    print("title:", title)
    print("rows:", [(k, S[k]["fps"], S[k]["quality"], S[k].get("d_quality"), S[k].get("d_quality_se"),
                     S[k].get("d_drift"), round(steps[k], 2)) for k in ARMS])
    return bad


if __name__ == "__main__":
    main()
