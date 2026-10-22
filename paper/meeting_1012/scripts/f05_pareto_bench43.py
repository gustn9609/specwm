#!/usr/bin/env python
"""f05_pareto_bench43 -- Bench-43 VLM quality / drift vs fps (two panels).

Our three operating modes (names used everywhere in the meeting package from 10/12):
    ours: teacher-parity mode = cw07_safe  (chunk-wise adaptive 0.7 + safe commit, facts.md section 12)
    ours: quality mode        = ours832    (832 2-step fused + partial commit)
    ours: realtime mode       = ours704    (704 2-step fused + partial commit)
The other chunk-wise arms (cw07_sep, cw07_pc1, cw05_pc1) are deliberately NOT drawn here
(kept uncluttered; they are in facts.md section 12).

Data:   paper/1012/bench43_summary.json  (keys teacher, student, sdvg07, sdvg22,
        mg2, wp, cw07_safe, ours832, ours704; fields fps, quality, quality_se, drift, drift_se,
        d_quality, d_quality_se, d_drift, d_drift_se)
        lingbot-world-v2/specwm_out/bench43/<prompt>/<arm>_s1234/**.mp4  (output sizes for the
        footnote; optional -- if the videos are gone the footnote falls back to the nominal sizes)
Option: paper/meeting_1012/data/fps_quiet.json  ({arm: fps}); used ONLY if it holds a value for
        all 9 arms (then every point is "quiet"). A partial file is refused, so quiet and
        under-load fps are never mixed on one axis. (The quiet timing pass, facts.md section 15, covers
        only jar + castle and not every arm, so the axis stays bench-43 under load; its teacher value
        4.08 fps and the quiet steady speed-ups of the three modes are read from facts.md section 15;
        the title leads with them (facts.md section 15: "prefer the quiet steady ratios in slides") and gives
        the bench-43 under-load ratio in parentheses. The teacher's older "4.9 fps quiet" is a whole-clip
        average, not used.)
Fixed text (not in the JSON): shared-GPU 704 fps 10.1 (facts.md section 6 / paper/1012/README.md W2),
        MG2 compile note and official 25 fps H100 claim (facts.md section 9), goal rule
        (>= 20 fps, quality >= 7.2; paper/1012/README.md sections T and U3, set 10/11 on the
        12-prompt set), teacher-parity mode recipe and mean steps 3.01 (facts.md section 12).
Style:  markers = light tint fill + solid edge in the arm colour; +-1 SE bars drawn above every
        marker, so the panel-a quality bars (SE 0.07-0.09, ~11 px) are visible inside the markers.
Output: paper/meeting_1012/figs/f05_pareto_bench43.{png,svg}

Run:    CUDA_VISIBLE_DEVICES="" venv/bin/python paper/meeting_1012/scripts/f05_pareto_bench43.py
"""
import argparse
import collections
import glob
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

ROOT = "/home/hk28965/specwm"
SUMMARY = os.path.join(ROOT, "paper/1012/bench43_summary.json")
FPS_QUIET = os.path.join(ROOT, "paper/meeting_1012/data/fps_quiet.json")
FACTS = os.path.join(ROOT, "paper/meeting_1012/facts.md")
VIDEO_ROOT = os.path.join(ROOT, "lingbot-world-v2/specwm_out/bench43")
OUT_DIR = os.path.join(ROOT, "paper/meeting_1012/figs")
FIG_ID = "f05_pareto_bench43"

# pre-set decision rule (README section T line "pre-set decision rule (A < 7.2 -> close)" and
# section U3 "No configuration reaches >= 20 fps and >= 7.2"), defined 10/11 on the 12-prompt set
GOAL_FPS = 20.0
GOAL_QUALITY = 7.2
# facts.md section 6 / README W2: ours 704 (realtime mode) on ONE shared GPU, cloud + full VAE serial = 10.1 fps
SHARED_GPU_704_FPS = 10.1

OURS = ["cw07_safe", "ours832", "ours704"]  # teacher-parity, quality, realtime mode
ARMS = ["teacher", "sdvg07", "sdvg22", "student", "mg2", "wp"] + OURS
SHORT = {
    "teacher": "teacher 14B",
    "student": "student 1.3B",
    "sdvg07": "SDVG τ −0.7",
    "sdvg22": "SDVG τ −2.2",
    "mg2": "Matrix-Game 2.0",
    "wp": "Waypoint-1.5-1B",
    "cw07_safe": "ours: teacher-parity mode",
    "ours832": "ours: quality mode",
    "ours704": "ours: realtime mode",
}
# legend text: mode name + recipe (the JSON 'label' of the ours arms is the old recipe-only name)
LEGEND = {
    "cw07_safe": "ours: teacher-parity mode (chunk-wise adaptive 0.7 + safe commit)",
    "ours832": "ours: quality mode (832 2-step, fused + partial commit)",
    "ours704": "ours: realtime mode (704 2-step, fused + partial commit)",
}
# nominal output size per arm (facts.md sections 1, 2, 9, 12). "≈" = fixed pixel area (max_area),
# aspect follows the start image. cw07_safe = 14B at the teacher's 832 area (videos: 848×464 on
# 31/43 prompts, same as the teacher).
RES = {
    "teacher": "≈832×480", "student": "≈832×480", "sdvg07": "≈832×480", "sdvg22": "≈832×480",
    "cw07_safe": "≈832×480", "ours832": "≈832×480", "ours704": "≈704×384", "mg2": "640×352",
    "wp": "1024×512",
}
GAME = {"mg2", "wp"}  # keyboard/mouse game models with converted camera inputs (facts.md section 9)
VIDEO_ARM = {"teacher": "uniform4", "ours704": "pc1_704"}  # for the most-common-size footnote

COLOR = {
    "teacher": "#1f1f1f",
    "student": "#e07b00",
    "sdvg07": "#8d6e63",
    "sdvg22": "#c49a6c",
    "mg2": "#2e7d32",
    "wp": "#00897b",
    "cw07_safe": "#1565c0",  # teacher-parity mode (new; lighter blues reserved for other cw arms)
    "ours832": "#7b4bd1",
    "ours704": "#d81b60",
}
MARKER = {a: "o" for a in ARMS}
MARKER.update({a: "*" for a in OURS})
MARKER.update({"teacher": "s", "sdvg07": "^", "sdvg22": "^", "mg2": "D", "wp": "D"})
# teacher drawn hollow and ON TOP of the SDVG tau -0.7 triangle (0.4 fps to its left, same quality)
MFC = {"teacher": "none"}
MEC = {"teacher": "#1f1f1f"}
MEW = {"teacher": 1.8}
ZORDER = {a: 4 for a in ARMS}
ZORDER.update({"teacher": 4.6})
ZORDER.update({a: 5 for a in OURS})
# Error bars are drawn in their own errorbar(fmt='none') call ABOVE every marker, and markers have a
# light tint fill with a solid edge, so a bar shorter than its marker stays visible (panel a:
# quality SE 0.07-0.09 is ~11-13 px at dpi 200, less than the radius of a star, the MG2 diamond
# or the SDVG triangle; with solid markers drawn on top those bars were hidden).
BAR_Z = 5.5
BAR_LW = 1.4
FILL_KEEP = 0.30  # marker fill = 30 % arm colour + 70 % white
MSIZE = {a: 9 for a in ARMS}
MSIZE.update({a: 13 for a in OURS})  # was 17: a 17 pt star (radius ~24 px) swallowed its bar
MSIZE.update({"mg2": 8, "wp": 8})

INK = "#222222"
MUTED = "#6b6b6b"
FOOT = "#7f7f7f"
GRID = "#e6e6e6"
GOAL_C = "#9e9e9e"

FIG_W, FIG_H = 12.0, 7.8  # slide-friendly; taller than 16:9 for the 3-line title, 3-row legend
#                           and the footer
LEG_NCOL = 3  # 3 x 3 legend: baselines in columns 1-2, the three ours modes in column 3

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.edgecolor": "#9a9a9a",
    "axes.linewidth": 0.8,
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "svg.fonttype": "none",
})


def load(fps_json):
    with open(SUMMARY) as f:
        s = json.load(f)
    missing = [a for a in ARMS if a not in s]
    if missing:
        sys.exit(f"missing arms in {SUMMARY}: {missing}")
    d = {a: dict(s[a]) for a in ARMS}
    quiet = False
    if fps_json and os.path.exists(fps_json):
        with open(fps_json) as f:
            q = json.load(f)
        have = [a for a in ARMS if q.get(a) is not None]
        if len(have) == len(ARMS):
            for a in ARMS:
                d[a]["fps"] = float(q[a])
            quiet = True
        else:
            lack = [a for a in ARMS if a not in have]
            print(f"[warn] {fps_json} lacks {lack}; partial quiet override refused, "
                  "all points stay under load")
    return d, quiet


def quiet_steady():
    """facts.md section 15 (quiet timing pass, steady s/chunk): teacher fps and the speed-up column of the
    three modes. Fails loudly if a row is missing."""
    txt = open(FACTS, encoding="utf-8").read()
    sec = txt.split("\n## 15.")[1]
    rows = {}
    for line in sec.splitlines():
        if line.startswith("|"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            rows[c[0]] = c
    key = {"teacher": "teacher 14B 4-step (uniform4)",
           "cw07_safe": "ours teacher-parity mode (chunk-wise 0.7, safe commit)",
           "ours832": "ours quality mode (832 fused + partial)",
           "ours704": "ours realtime mode (704 fused + partial)"}
    return {a: (rows[k][3], rows[k][4]) for a, k in key.items()}  # (fps, speed-up) as written in facts.md


def common_size(arm_dir):
    """Most common (w, h) over seed-1234 videos of one bench-43 arm; None if videos are missing."""
    import cv2  # local import: only needed for the footnote
    c = collections.Counter()
    for sc in sorted(os.listdir(VIDEO_ROOT)) if os.path.isdir(VIDEO_ROOT) else []:
        fs = sorted(glob.glob(os.path.join(VIDEO_ROOT, sc, f"{arm_dir}_s1234", "**", "*.mp4"),
                              recursive=True))
        if not fs:
            continue
        v = cv2.VideoCapture(fs[0])
        w, h = int(v.get(3)), int(v.get(4))
        v.release()
        if w and h:
            c[(w, h)] += 1
    if not c:
        return None
    (w, h), k = c.most_common(1)[0]
    return w, h, k, sum(c.values())


# --------------------------------------------------------------------------- label placement
def _overlap(b1, b2, pad=0.0):
    return not (b1.x1 + pad <= b2.x0 or b2.x1 + pad <= b1.x0 or
                b1.y1 + pad <= b2.y0 or b2.y1 + pad <= b1.y0)


def _label_style(arm):
    bold = arm in OURS
    return dict(fontsize=10, color=COLOR[arm] if bold else INK,
                fontweight="bold" if bold else "normal", zorder=6)


def tint(c, keep):
    """Arm colour blended toward white; keep = share of the colour retained."""
    return tuple(1 - (1 - v) * keep for v in to_rgb(c))


def mstyle(arm):
    """Marker style shared by the panels, the inset and the legend: light tint fill + solid edge
    in the arm colour (teacher: hollow, black edge)."""
    return dict(marker=MARKER[arm], mfc=MFC.get(arm, tint(COLOR[arm], FILL_KEEP)),
                mec=MEC.get(arm, COLOR[arm]), mew=MEW.get(arm, 1.3))


def draw_point(ax, arm, x, y, se):
    """Marker first, then its +-1 SE bar in a separate errorbar(fmt='none') call ABOVE every
    marker; the solid bar reads against the tinted fill even when it is shorter than the marker."""
    ax.plot([x], [y], ls="none", ms=MSIZE[arm], zorder=ZORDER[arm], **mstyle(arm))
    ax.errorbar([x], [y], yerr=se, fmt="none", ecolor=COLOR[arm], elinewidth=BAR_LW,
                capsize=3, capthick=BAR_LW, zorder=BAR_Z)


def place_labels(fig, ax, pts, prefs, hard_extra=(), skip=()):
    """Greedy collision-free placement of one text label per point.

    pts: list of (arm, x, y, yerr). prefs: arm -> preferred angle (deg) or a fixed
    (dx, dy, leader, ha) offset in points. Hard obstacles: other labels, markers, error bars,
    the axes edge, and hard_extra bboxes (reference lines, texts, inset).
    skip: arms that stay obstacles but get no label here (labelled elsewhere).
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    to_disp = ax.transData.transform
    dpi_scale = fig.dpi / 72.0

    hard = list(hard_extra)
    for arm, x, y, yerr in pts:
        cx, cy = to_disp((x, y))
        rad = MSIZE[arm] * 0.5 * dpi_scale + 1
        hard.append(Bbox([[cx - rad, cy - rad], [cx + rad, cy + rad]]))
        (ex, ey0), (_, ey1) = to_disp((x, y - yerr)), to_disp((x, y + yerr))
        hard.append(Bbox([[ex - 4, ey0], [ex + 4, ey1]]))
    axbox = ax.get_window_extent(r)
    axbox = Bbox([[axbox.x0 + 2, axbox.y0 + 2], [axbox.x1 - 2, axbox.y1 - 2]])

    def fits(bbp):
        return axbox.contains(bbp.x0, bbp.y0) and axbox.contains(bbp.x1, bbp.y1)

    placed = []
    radii = [9, 14, 22, 32, 44, 58]
    for arm, x, y, yerr in pts:
        if arm in skip:
            continue
        pref = prefs.get(arm, 45)
        if isinstance(pref, tuple):  # fixed offset (dx, dy) in points, leader line, ha
            dx, dy, leader, ha = pref
            va = "bottom" if dy > 3 else ("top" if dy < -3 else "center")
            t = ax.annotate(SHORT[arm], (x, y), xytext=(dx, dy), textcoords="offset points",
                            ha=ha, va=va, **_label_style(arm))
            bbp = t.get_window_extent(r)
            clash = [o for o in hard if _overlap(bbp, o)] + [o for o in placed if _overlap(bbp, o, 2)]
            if clash or not fits(bbp):
                print(f"[warn] fixed label for {arm} collides with {len(clash)} boxes or the axes edge")
            if leader:  # from the bottom/top centre of the label to the point
                px, py = to_disp((x, y))
                lx = (0.5 * (bbp.x0 + bbp.x1) - px) / dpi_scale
                ly = ((bbp.y0 if dy > 0 else bbp.y1) - py) / dpi_scale
                ax.annotate("", (x, y), xytext=(lx, ly), textcoords="offset points",
                            arrowprops=dict(arrowstyle="-", color="#9a9a9a", lw=0.8,
                                            shrinkA=1, shrinkB=MSIZE[arm] * 0.55), zorder=2)
            placed.append(bbp)
            continue
        best = None
        angles = sorted(range(0, 360, 15), key=lambda a: min(abs(a - pref), 360 - abs(a - pref)))
        for ri, rad in enumerate(radii):
            for ai, ang in enumerate(angles):
                dx = rad * math.cos(math.radians(ang))
                dy = rad * math.sin(math.radians(ang))
                ha = "left" if dx > 3 else ("right" if dx < -3 else "center")
                va = "bottom" if dy > 3 else ("top" if dy < -3 else "center")
                t = ax.annotate(SHORT[arm], (x, y), xytext=(dx, dy), textcoords="offset points",
                                ha=ha, va=va, **_label_style(arm))
                bb = t.get_window_extent(r)
                bbp = Bbox([[bb.x0 - 2, bb.y0 - 1], [bb.x1 + 2, bb.y1 + 1]])
                ok = fits(bbp) and not any(_overlap(bbp, ob) for ob in hard) \
                    and not any(_overlap(bbp, ob, pad=2) for ob in placed)
                score = ri * 10 + ai * 0.6
                if ok and (best is None or score < best[0]):
                    if best is not None:
                        best[1].remove()
                    best = (score, t, bbp, rad)
                else:
                    t.remove()
            if best is not None and best[0] < (ri + 1) * 10:
                break
        if best is None:
            print(f"[warn] no free spot for {arm}; using a fixed offset")
            t = ax.annotate(SHORT[arm], (x, y), xytext=(12, 12), textcoords="offset points",
                            **_label_style(arm))
            best = (0, t, t.get_window_extent(r), 12)
        _, t, bbp, rad = best
        if rad >= 22:
            ax.annotate("", (x, y), xytext=t.xyann, textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", color="#9a9a9a", lw=0.8,
                                        shrinkA=1, shrinkB=MSIZE[arm] * 0.55), zorder=2)
        placed.append(bbp)
    return placed


def wrap_measured(fig, sentences, fontsize, max_px):
    """Greedy word wrap of the joined sentences using rendered text width."""
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


# --------------------------------------------------------------------------- figure
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps-json", default=FPS_QUIET,
                    help="optional {arm: fps} quiet override (used only if it covers all 9 arms)")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    d, quiet = load(args.fps_json)
    cond_short = "quiet" if quiet else "under load"

    fps_max = max(d[a]["fps"] for a in ARMS)
    xmax = max(30.0, math.ceil((fps_max + 4.5) / 2) * 2)  # room for a centred label above the fastest point
    n = d["teacher"].get("clips", 86)

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    fig.canvas.draw()

    # ---- footer (measured wrap), then the vertical layout from the bottom up
    try:
        sz832 = common_size(VIDEO_ARM["teacher"])
        sz704 = common_size(VIDEO_ARM["ours704"])
    except Exception as e:  # cv2 missing etc.
        print(f"[warn] output sizes not read from videos: {e}")
        sz832 = sz704 = None
    if sz832 and sz704:
        size_note = (f"mostly {sz832[0]}×{sz832[1]}, {sz832[2]}/{sz832[3]} prompts; "
                     f"{sz704[0]}×{sz704[1]}, {sz704[2]}/{sz704[3]}")
    else:
        size_note = "exact size varies by prompt"
    if quiet:
        rel = os.path.relpath(args.fps_json, ROOT)
        rel = os.path.basename(args.fps_json) if rel.startswith("..") else rel
        load_note = f"measured quiet (no other GPU jobs; {rel})."
    else:
        qs = quiet_steady()
        load_note = (f"measured under load (7 GPU jobs at once; teacher {d['teacher']['fps']:.1f} fps "
                     "here).")
    def sgn(v, nd):  # signed number with a typographic minus
        return f"{v:+.{nd}f}".replace("-", "−")

    def dq(a):  # paired delta vs the teacher, from the JSON (quality / drift)
        r = d[a]
        return (f"{sgn(r['d_quality'], 2)} ± {r['d_quality_se']:.2f} / "
                f"{sgn(r['d_drift'], 1)} ± {r['d_drift_se']:.1f}")

    speed_note = (("title speed-ups come from the quiet steady timing pass (facts.md §15: jar + sandcastle, generation only; "
                   f"teacher {qs['teacher'][0]} fps); bench-43 under-load ratios in parentheses.") if not quiet
                  else "speed-ups are ratios of quiet fps.")
    sentences = [
        f"Bench-43: 38 StEvo camera-WM tasks + 5 easy scenes × 2 seeds = {n} clips per arm; Qwen3-VL-32B judge "
        "(8 frames, 448 px); error bars ±1 SE over clips. Drift prompt: AUROC 0.82 vs human "
        "severe-drift labels; quality prompt NOT validated against humans.",
        "fps = 12 / steady s per chunk on one RTX PRO 6000 Blackwell (not H100), " + load_note
        + " The " + speed_note,
        "Teacher-parity mode = 14B 4-step schedule; a learned head stops each chunk once ≥ 70 % of "
        "tokens are predicted converged (mean 3.01 steps); safe commit = partial commit if the chunk "
        "stopped at step ≥ 3, else a full clean pass. "
        f"Paired Δ vs teacher ({d['cw07_safe'].get('paired_n', n)} clips, quality / drift): "
        f"teacher-parity {dq('cw07_safe')}; quality mode {dq('ours832')}; "
        f"realtime mode {dq('ours704')}.",
        "LingBot arms (teacher, student, SDVG, all ours modes) exclude the final VAE decode; the "
        "quality and realtime modes assume it runs on a separate device GPU (on one shared GPU, "
        f"cloud + full VAE = {SHARED_GPU_704_FPS} fps for the realtime mode). MG2 and Waypoint "
        "include their own decoder; MG2 run without torch.compile (SDPA shim; official claim "
        "25 fps on H100).",
        "† keyboard/mouse game models, camera paths converted to mouse/WASD inputs. "
        f"≈ : fixed pixel area, aspect follows the start image ({size_note}).",
        f"Shaded goal = pre-set decision rule from 10/11 on the 12-prompt set "
        "(paper/1012/README.md §T, §U3). Data: paper/1012/bench43_summary.json; "
        "teacher-parity recipe: paper/meeting_1012/facts.md §12.",
    ]
    foot_lines = wrap_measured(fig, sentences, 9, 0.95 * FIG_W * fig.dpi)
    lh = 9 * 1.4 / 72.0  # inches per footer line
    foot_h = len(foot_lines) * lh
    leg_rows = math.ceil(len(ARMS) / LEG_NCOL)
    legend_top_in = 0.06 + foot_h + 0.08 + 0.46 + 0.21 * (leg_rows - 2)
    ax_bottom_in = legend_top_in + 0.62

    # ---- title: the finding, computed from the data; one line per mode, "fps under load" stated
    # once in line 1. Drawn before the axes so the axes top follows the measured title height.
    t, p, s, m, mg = d["teacher"], d["cw07_safe"], d["ours832"], d["ours704"], d["mg2"]
    fps_cond = "fps under load" if not quiet else "quiet fps"
    if quiet:
        sp_p = f"{p['fps'] / t['fps']:.2f}× the teacher's {fps_cond} ({p['fps']:.1f} vs {t['fps']:.1f})"
        sp_s = f"{s['fps'] / t['fps']:.1f}× the teacher's fps"
    else:  # quiet steady ratio first (facts.md section 15), bench-43 under-load ratio in parentheses
        sp_p = f"{qs['cw07_safe'][1]} the teacher's speed quiet ({p['fps'] / t['fps']:.2f}× under load)"
        sp_s = f"{qs['ours832'][1]} quiet ({s['fps'] / t['fps']:.1f}× under load)"
    title = "\n".join([
        f"Ours, teacher-parity mode: {sp_p} at quality {sgn(p['d_quality'], 2)} ± "
        f"{p['d_quality_se']:.2f} (paired);",
        f"quality mode: {sp_s} at quality "
        f"{sgn(s['d_quality'], 2)} ± {s['d_quality_se']:.2f} (paired);",
        # drift difference printed at the same 1 decimal as every drift value (70.0 - 56.9 = 13.1)
        f"realtime mode matches Matrix-Game 2.0 quality ({m['quality']:.2f} vs "
        f"{mg['quality']:.2f}) with {mg['drift'] - m['drift']:.1f} points less drift, at "
        f"{m['fps']:.1f} vs {mg['fps']:.1f} fps",
    ])
    ttl = fig.text(0.065, 1 - 0.22 / FIG_H, title, ha="left", va="top", fontsize=13, color=INK,
                   fontweight="bold", linespacing=1.35)
    rr0 = fig.canvas.get_renderer()
    for fs in (13, 12.5, 12, 11.5, 11):  # shrink only if a line would run off the canvas
        ttl.set_fontsize(fs)
        if ttl.get_window_extent(rr0).x1 <= fig.bbox.width - 12:
            break
    else:
        print("[warn] title wider than the canvas even at 11 pt")
    print("title font size:", ttl.get_fontsize())
    title_bottom_in = FIG_H - ttl.get_window_extent(rr0).y0 / fig.dpi
    ax_top_in = FIG_H - (title_bottom_in + 0.50)  # 0.50 in for the panel titles a / b
    yb, hh = ax_bottom_in / FIG_H, (ax_top_in - ax_bottom_in) / FIG_H

    axL = fig.add_axes([0.065, yb, 0.41, hh])
    axR = fig.add_axes([0.575, yb, 0.41, hh])

    for ax in (axL, axR):
        ax.set_facecolor("white")
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlim(0, xmax)
        ax.set_xlabel(f"Throughput, fps ({cond_short}; see footer)   ↑ higher is better")

    # ---- left: quality
    qlo, qhi = 3.6, 8.1
    axL.set_ylim(qlo, qhi)
    axL.set_ylabel("VLM visual quality (1–10)   ↑ higher is better")
    axL.set_title("a   Visual quality vs speed", loc="left", fontsize=11, color=INK, pad=8)
    axL.add_patch(Rectangle((GOAL_FPS, GOAL_QUALITY), xmax - GOAL_FPS, qhi - GOAL_QUALITY,
                            facecolor="#f3f3f3", edgecolor="none", zorder=0.5))
    axL.axvline(GOAL_FPS, color=GOAL_C, ls=(0, (5, 3)), lw=1.2, zorder=1)
    axL.axhline(GOAL_QUALITY, color=GOAL_C, ls=(0, (5, 3)), lw=1.2, zorder=1)
    gtxt = axL.text(GOAL_FPS + 0.4, qhi - 0.05, f"goal (pre-set rule):\n≥ {GOAL_FPS:.0f} fps, "
                    f"quality ≥ {GOAL_QUALITY}", ha="left", va="top", fontsize=9, color=MUTED,
                    zorder=3, linespacing=1.25)
    q_txt = axL.text(xmax - 0.3, GOAL_QUALITY - 0.04, f"quality {GOAL_QUALITY}", ha="right",
                     va="top", fontsize=9, color=MUTED, zorder=3)
    f_txt = axL.text(GOAL_FPS + 0.3, qlo + 0.05, f"{GOAL_FPS:.0f} fps", ha="left", va="bottom",
                     fontsize=9, color=MUTED, zorder=3)

    # ---- right: drift
    dlo, dhi = 40, 81  # room below the teacher-parity star (drift 47.7 ± 3.2) for labels
    axR.set_ylim(dlo, dhi)
    axR.set_ylabel("VLM drift score (0–100)   ↓ lower is better")
    axR.set_title("b   Drift vs speed", loc="left", fontsize=11, color=INK, pad=8)
    axR.axvline(GOAL_FPS, color=GOAL_C, ls=(0, (5, 3)), lw=1.2, zorder=1)
    f_txtR = axR.text(GOAL_FPS - 0.3, dhi - 0.2, f"{GOAL_FPS:.0f} fps goal", ha="right", va="top",
                      fontsize=9, color=MUTED, zorder=3)
    b_txtR = axR.text(xmax - 0.3, dlo + 0.4, "better ↘", ha="right", va="bottom", fontsize=9,
                      color=MUTED, zorder=3)

    ptsL, ptsR = [], []
    for a in ARMS:
        r = d[a]
        draw_point(axL, a, r["fps"], r["quality"], r["quality_se"])
        draw_point(axR, a, r["fps"], r["drift"], r["drift_se"])
        ptsL.append((a, r["fps"], r["quality"], r["quality_se"]))
        ptsR.append((a, r["fps"], r["drift"], r["drift_se"]))

    # ---- panel a inset: zoom on teacher vs SDVG tau -0.7 when they overlap at this scale
    # (bench-43: same quality 7.51, 0.4 fps apart). The box drawn in the main panel is exactly
    # the inset's data range; each label sits on the outer side of its own marker so neither
    # label spans the other marker. If a re-run moves them apart, they get normal labels.
    s7, tt = d["sdvg07"], d["teacher"]
    use_zoom = abs(tt["fps"] - s7["fps"]) < 1.0 and abs(tt["quality"] - s7["quality"]) < 0.15
    axI = None
    if use_zoom:
        lo, hi = sorted(("sdvg07", "teacher"), key=lambda a: d[a]["fps"])
        gap = d[hi]["fps"] - d[lo]["fps"]
        width = max(gap / 0.14, 2.5)  # ~43 % of the inset width left of lo / right of hi for labels
        side = (width - gap) / 2
        qc = 0.5 * (tt["quality"] + s7["quality"])
        zx0, zx1, zy0, zy1 = d[lo]["fps"] - side, d[hi]["fps"] + side, qc - 0.21, qc + 0.21
        axL.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, facecolor="none",
                                edgecolor="#9a9a9a", lw=0.8, zorder=3))
        axI = axL.inset_axes([0.035, 0.05, 0.46, 0.28])
        axI.set_xlim(zx0, zx1)
        axI.set_ylim(zy0, zy1)
        axI.set_xticks([])
        axI.set_yticks([])
        for sp in axI.spines.values():
            sp.set_edgecolor("#9a9a9a")
            sp.set_linewidth(0.8)
        axI.set_facecolor("white")
        for a in ("sdvg07", "teacher"):
            r = d[a]
            draw_point(axI, a, r["fps"], r["quality"], r["quality_se"])
        for a, dx, ha in ((lo, -9, "right"), (hi, 9, "left")):
            r = d[a]
            axI.annotate(f"{SHORT[a]}\n{r['fps']:.1f} fps, {r['quality']:.2f}",
                         (r["fps"], r["quality"]), xytext=(dx, 0), textcoords="offset points",
                         ha=ha, va="center", fontsize=9, color=INK, linespacing=1.2)
        axI.set_title("zoom of the boxed region", fontsize=9, color=MUTED, pad=3, loc="left")

    # placement order: crowded first
    order = ["teacher", "sdvg07", "cw07_safe", "ours832", "sdvg22", "ours704", "mg2", "student",
             "wp"]
    ptsL.sort(key=lambda p: order.index(p[0]))
    ptsR.sort(key=lambda p: order.index(p[0]))
    # realtime mode in panel a: right-aligned under its star (a centred label would cross the
    # 20 fps line). Panel b: teacher label below-left of the square (the teacher-parity star sits
    # below-right of it), teacher-parity label centred under its own error bar.
    prefL = {"cw07_safe": 270, "ours832": 30, "sdvg22": 270,
             "ours704": (43, -12, False, "right"), "mg2": 315, "student": 270, "wp": 90}
    prefR = {"teacher": (22, -18, True, "right"), "sdvg07": (-6, 24, True, "center"),
             "cw07_safe": 270, "ours832": 0, "sdvg22": 60, "ours704": 90, "mg2": 180,
             "student": 120, "wp": 90}

    fig.canvas.draw()
    rr = fig.canvas.get_renderer()

    def bb(t, pad=1):
        b = t.get_window_extent(rr)
        return Bbox([[b.x0 - pad, b.y0 - pad], [b.x1 + pad, b.y1 + pad]])

    def line_boxes(ax, vlines=(), hlines=()):
        out = []
        ab = ax.get_window_extent(rr)
        for xv in vlines:
            x0, _ = ax.transData.transform((xv, 0))
            out.append(Bbox([[x0 - 4, ab.y0], [x0 + 4, ab.y1]]))
        for yv in hlines:
            _, y0 = ax.transData.transform((0, yv))
            out.append(Bbox([[ab.x0, y0 - 4], [ab.x1, y0 + 4]]))
        return out

    hardL = [bb(gtxt), bb(q_txt), bb(f_txt)] + line_boxes(axL, [GOAL_FPS], [GOAL_QUALITY])
    if use_zoom:
        # teacher and SDVG tau -0.7 overlap at this scale: one group label above the zoom box,
        # individual labels inside the inset
        grp = axL.annotate(f"{SHORT['teacher']} + {SHORT['sdvg07']}", (zx0, zy1),
                           xytext=(0, 3), textcoords="offset points", ha="left", va="bottom",
                           fontsize=10, color=INK, zorder=6)
        if any(_overlap(bb(grp), o) for o in line_boxes(axL, [GOAL_FPS], [GOAL_QUALITY])):
            print("[warn] group label crosses a goal line")
        zoom_box = Bbox([axL.transData.transform((zx0, zy0)), axL.transData.transform((zx1, zy1))])
        hardL += [zoom_box, axI.get_tightbbox(rr), bb(grp, 2)]
        skipL = ("teacher", "sdvg07")
    else:
        prefL.update({"teacher": 60, "sdvg07": 150})
        skipL = ()
    place_labels(fig, axL, ptsL, prefL, hard_extra=hardL, skip=skipL)
    place_labels(fig, axR, ptsR, prefR,
                 hard_extra=[bb(f_txtR), bb(b_txtR)] + line_boxes(axR, [GOAL_FPS]))

    # ---- legend (full arm names + output size) above the footer
    def leg_label(a):
        base = LEGEND.get(a, d[a].get("label", SHORT[a]))
        return f"{base}{' †' if a in GAME else ''} · {RES[a]}"

    # column-major fill with 3 columns: baselines in columns 1-2, the three ours modes in column 3
    leg_order = ["teacher", "student", "sdvg07", "sdvg22", "mg2", "wp"] + OURS
    handles = [Line2D([0], [0], color="none", ms=MSIZE[a] * (1.0 if a in OURS else 0.95),
                      label=leg_label(a), **mstyle(a))
               for a in leg_order]
    leg = fig.legend(handles=handles, loc="upper center",
                     bbox_to_anchor=(0.5, legend_top_in / FIG_H), ncol=LEG_NCOL,
                     frameon=False, fontsize=9.5, handletextpad=0.4, columnspacing=1.4)
    fig.text(0.5, 0.06 / FIG_H, "\n".join(foot_lines), ha="center", va="bottom", fontsize=9,
             color=FOOT, linespacing=1.4)

    # ---- sanity: nothing outside the canvas, legend clear of the x labels
    fig.canvas.draw()
    W, H = fig.bbox.width, fig.bbox.height
    for name, art in (("title", ttl), ("legend", leg)):
        b = art.get_window_extent(rr)
        if b.x0 < 5 or b.x1 > W - 5 or b.y0 < 0 or b.y1 > H:
            print(f"[warn] {name} runs off the canvas: {b}")
    lb = leg.get_window_extent(rr)
    for ax in (axL, axR):
        xl = ax.xaxis.label.get_window_extent(rr)
        if _overlap(lb, xl):
            print("[warn] legend overlaps an x-axis label")

    os.makedirs(args.out_dir, exist_ok=True)
    png = os.path.join(args.out_dir, FIG_ID + ".png")
    svg = os.path.join(args.out_dir, FIG_ID + ".svg")
    fig.savefig(png, dpi=200, facecolor="white")
    fig.savefig(svg, facecolor="white")
    print("wrote", png)
    print("wrote", svg)
    print("fps condition:", cond_short)


if __name__ == "__main__":
    main()
