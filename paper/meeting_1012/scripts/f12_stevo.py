#!/usr/bin/env python
"""f12_stevo: StEvo-Bench subset (38 camera-WM tasks, seed 1234, local Qwen3-VL-32B judge) per arm.

Re-run:  CUDA_VISIBLE_DEVICES="" /home/hk28965/specwm/venv/bin/python \
         /home/hk28965/specwm/paper/meeting_1012/scripts/f12_stevo.py

Data sources (paths relative to /home/hk28965/specwm):
  - lingbot-world-v2/specwm_out/stevo38/eval_runs/lingbot_b38_<arm>/summary.json
    (key tasks[].llm_evals["openai__qwen3-vl-32b"]: per-task booleans; overall: the means)
    arms: teacher, student, sdvg07, mg2, wp, ours832, ours704
  - paper/meeting_1012/facts.md section 14 (canonical table; every fraction drawn here is asserted to
    match it to rounding)
Outputs: figs/f12_stevo.{png,svg}; data/f12_stevo.json (counts, fractions, Wilson CIs, paired tests);
data/f12_stevo_table.md. No GPU is used.

Panel a: physical inaccuracy (share of the 38 tasks flagged, lower is better) per arm, sorted, with
Wilson 95 % score intervals (n = 38). Panel b: small multiples of the other five criteria on the same
rows. Paired comparisons (same 38 tasks): exact two-sided McNemar test (binomial on the discordant tasks),
no multiple-comparison correction.
Checks before drawing: all arms scored on the same 38 task ids; per-task booleans reproduce the
summary.json overall means and the facts.md section 14 table; control_success == occlusion_done AND
trigger_applied and task_success == state_evol AND NOT physical_inaccuracy on every task.
"""
import json
import math
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402

ROOT = "/home/hk28965/specwm"
PKG = os.path.join(ROOT, "paper/meeting_1012")
FACTS = os.path.join(PKG, "facts.md")
EVAL = os.path.join(ROOT, "lingbot-world-v2/specwm_out/stevo38/eval_runs")
OUT_FIG = os.path.join(PKG, "figs")
OUT_JSON = os.path.join(PKG, "data/f12_stevo.json")
OUT_TAB = os.path.join(PKG, "data/f12_stevo_table.md")
JUDGE = "openai__qwen3-vl-32b"
Z = 1.959964  # two-sided 95 %
NB = " "  # no-break space

# ---------------------------------------------------------------- style (meeting colours)
C = {"teacher": "#1f1f1f", "student": "#e07b00", "sdvg07": "#8d6e63", "mg2": "#2e7d32", "wp": "#00897b",
     "ours832": "#7b4bd1", "ours704": "#d81b60"}
INK, INK2, MUTED, GRID = "#1f1f1f", "#4a4a4a", "#7a7a7a", "#e6e6e6"
BAND = "#f2f2f2"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
    "axes.edgecolor": "#9a9a9a", "axes.linewidth": 0.8,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "xtick.color": "#555555", "ytick.color": "#555555",
})

ARMS = ["teacher", "student", "sdvg07", "mg2", "wp", "ours832", "ours704"]
NAME = {"teacher": "teacher 14B", "student": "student 1.3B", "sdvg07": "SDVG τ −0.7",
        "mg2": "Matrix-Game 2.0", "wp": "Waypoint-1.5-1B", "ours832": "ours: quality mode",
        "ours704": "ours: realtime mode"}
SHORT = {"teacher": "teacher", "student": "student", "sdvg07": "SDVG τ −0.7", "mg2": "Matrix-Game 2.0",
         "wp": "Waypoint", "ours832": "ours 832", "ours704": "ours 704"}
DESC = {"teacher": "official causal-fast, 4-step", "student": "official causal-fast, 4-step",
        "sdvg07": "1.3B draft + 14B target, paper τ", "mg2": "640×352, keyboard/mouse model",
        "wp": "1024×512, keyboard/mouse model", "ours832": "832, 14B 2-step, fused + partial",
        "ours704": "≈704×384, 14B 2-step, fused + partial"}
B14 = ("teacher", "sdvg07", "ours832", "ours704")  # 14B-based arms (14B target / 14B cloud model)
MODES = ("ours832", "ours704")  # our operating modes that were run here (teacher-parity: not run)
# table name in facts.md section 14 -> arm key
FACT_ROW = {"teacher 14B 4-step": "teacher", "student 1.3B 4-step": "student", "SDVG + actions τ −0.7": "sdvg07",
            "Matrix-Game 2.0": "mg2", "Waypoint-1.5-1B": "wp", "ours quality mode (832)": "ours832",
            "ours realtime mode (704)": "ours704"}
# metric key in llm_evals, key in summary overall, label, short definition, better
METRICS = [
    ("physical_inaccuracy", "avg_physical_inaccuracy", "physical inaccuracy", "a physical inaccuracy was detected", "lo"),
    ("occlusion_done", "avg_occlusion_done", "occlusion done", "the look-away happened", "hi"),
    ("trigger_applied", "avg_trigger_applied", "trigger applied", "the prompted action happened", "hi"),
    ("control_success", "avg_control_success", "control success", "occlusion AND trigger", "hi"),
    ("state_evol", "avg_state_evol_success", "state evolution", "change visible after looking back", "hi"),
    ("task_success", "avg_task_success", "task success", "state evol. AND NOT phys. inacc.", "hi"),
]
FACT_COLS = ["occlusion_done", "trigger_applied", "physical_inaccuracy", "control_success", "state_evol",
             "task_success"]  # column order of the facts.md section 14 table


def wilson(k, n, z=Z):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b, c):
    """two-sided exact McNemar p-value: binomial(b + c, 0.5) on the discordant pairs"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def fmt_p(p, op="="):
    if p < 0.001:
        return "p < 0.001"
    return f"p {op} {p:.3f}" if p < 0.1 else f"p {op} {p:.2f}"


def lighten(hex_c, f=0.55):
    r, g, b = to_rgb(hex_c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


# ---------------------------------------------------------------- load + checks
T, OVERALL, NTASK = {}, {}, {}
for a in ARMS:
    s = json.load(open(os.path.join(EVAL, f"lingbot_b38_{a}", "summary.json"), encoding="utf-8"))
    T[a] = {t["task_id"]: t["llm_evals"][JUDGE] for t in s["tasks"]}
    OVERALL[a] = s["overall"][JUDGE]
    NTASK[a] = s["num_tasks"]
ids = sorted(T["teacher"])
assert all(sorted(T[a]) == ids for a in ARMS), "arms were scored on different task sets"
N = len(ids)
assert N == 38 and all(NTASK[a] == N and OVERALL[a]["num_tasks"] == N for a in ARMS)
for a in ARMS:
    for i in ids:
        e = T[a][i]
        assert e["control_success"] == (e["occlusion_done"] and e["trigger_applied"]), (a, i)
        assert e["task_success"] == (e["state_evol"] and not e["physical_inaccuracy"]), (a, i)

R = {a: {} for a in ARMS}
for a in ARMS:
    for m, ok, _, _, _ in METRICS:
        k = sum(bool(T[a][i][m]) for i in ids)
        assert abs(k / N - OVERALL[a][ok]) < 1e-9, (a, m)
        lo, hi = wilson(k, N)
        R[a][m] = dict(k=k, n=N, frac=k / N, lo=lo, hi=hi)

# facts.md section 14 table == summary.json (2-decimal rounding)
sec = open(FACTS, encoding="utf-8").read().split("## 14.")[1].split("## 15.")[0]
seen = set()
for line in sec.splitlines():
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells) == 7 and cells[0] in FACT_ROW:
        a = FACT_ROW[cells[0]]
        seen.add(a)
        for m, v in zip(FACT_COLS, cells[1:]):
            assert abs(round(R[a][m]["frac"], 2) - float(v)) < 1e-9, (a, m, R[a][m]["frac"], v)
assert seen == set(ARMS), seen
print("facts.md §14 matches the 7 summary.json files")

# ---------------------------------------------------------------- paired tests (physical inaccuracy)
PI = "physical_inaccuracy"


def paired(a, b, m=PI):
    only_a = sum(bool(T[a][i][m]) and not T[b][i][m] for i in ids)
    only_b = sum(bool(T[b][i][m]) and not T[a][i][m] for i in ids)
    return dict(only_a=only_a, only_b=only_b, p=mcnemar_exact(only_a, only_b))


PAIRS = {f"{a}_vs_{b}": paired(a, b) for a in B14 for b in ("mg2", "wp", "student")}
PAIRS.update({f"{a}_vs_teacher": paired(a, "teacher") for a in ("ours832", "ours704", "sdvg07")})
P = PAIRS
f2 = lambda x: f"{x:.2f}"  # noqa: E731

# claims written in the title / subtitle, checked here
pi = {a: R[a][PI]["frac"] for a in ARMS}
b14_lo, b14_hi = min(pi[a] for a in B14), max(pi[a] for a in B14)
assert b14_hi < pi["mg2"] < pi["wp"] < pi["student"]
assert all(P[f"{a}_vs_student"]["p"] < 0.01 for a in B14)
sig_wp = [a for a in B14 if P[f"{a}_vs_wp"]["p"] < 0.05]
sig_mg2 = [a for a in B14 if P[f"{a}_vs_mg2"]["p"] < 0.05]
assert sig_mg2 == ["ours832"], sig_mg2  # wording below assumes only ours 832 separates from MG2
assert set(sig_wp) == {"teacher", "ours832", "ours704"}, sig_wp
assert all(P[f"{a}_vs_teacher"]["p"] > 0.05 for a in ("ours832", "ours704", "sdvg07"))
se_max = max(R[a]["state_evol"]["frac"] for a in ARMS)
se_kmax = max(R[a]["state_evol"]["k"] for a in ARMS)
ts_kmax = max(R[a]["task_success"]["k"] for a in ARMS)

title = (f"StEvo-38: the 14B-based arms incl. ours ({f2(b14_lo)}–{f2(b14_hi)}{NB}↓) are flagged for physical inaccuracy "
         f"less often than the student ({f2(pi['student'])}) and Waypoint ({f2(pi['wp'])}); only ours quality mode "
         f"separates from Matrix-Game{NB}2.0 ({f2(pi['mg2'])}); state evolution is at the floor (0–{f2(se_max)}) for every arm")
max_p = lambda arms, b: max(P[f"{a}_vs_{b}"]["p"] for a in arms)  # noqa: E731
nonsig_mg2 = [a for a in B14 if a not in sig_mg2]
q832 = P["ours832_vs_teacher"]
q704 = P["ours704_vs_teacher"]
sub = [
    (f"Paired on the same {N} tasks (exact McNemar, uncorrected): every 14B-based arm is flagged less often than the "
     f"student (all {fmt_p(max_p(B14, 'student'), '≤')}) and, except SDVG ({fmt_p(P['sdvg07_vs_wp']['p'])}), than "
     f"Waypoint (all {fmt_p(max_p(sig_wp, 'wp'), '≤')}). Against Matrix-Game{NB}2.0 only ours quality mode separates "
     f"({R['ours832'][PI]['k']} vs {R['mg2'][PI]['k']} flagged, discordant {P['ours832_vs_mg2']['only_a']} vs "
     f"{P['ours832_vs_mg2']['only_b']}, {fmt_p(P['ours832_vs_mg2']['p'])}; "
     + ", ".join(f"{('SDVG' if a == 'sdvg07' else SHORT[a])} {fmt_p(P[f'{a}_vs_mg2']['p'])}" for a in nonsig_mg2) + ")."),
    (f"Ours vs teacher: quality mode {f2(pi['ours832'])} vs {f2(pi['teacher'])} (discordant {q832['only_a']} vs "
     f"{q832['only_b']}, {fmt_p(q832['p'])}), realtime mode {f2(pi['ours704'])} = {f2(pi['teacher'])} "
     f"({q704['only_a']} vs {q704['only_b']}) → teacher level, not better. State evolution and task success: "
     f"≤{NB}{max(se_kmax, ts_kmax)} of {N} tasks for every arm. Matrix-Game{NB}2.0's 'occlusion done' "
     f"{f2(R['mg2']['occlusion_done']['frac'])} = game-style mouse look that turns away visibly; the prompted action "
     f"rarely shows (trigger {f2(R['mg2']['trigger_applied']['frac'])})."),
]

foot = [
    f"Protocol: StEvo-Bench (arXiv 2603.13215) subset = the {N} camera-WM tasks of bench-43, seed 1234, one video per task "
    "and arm; 249 frames for every LingBot arm and Matrix-Game 2.0 (MG2 needs latent frames divisible by 3), "
    "256 for Waypoint (4 frames per call), covering the 253-frame look-away-and-back camera path.",
    "Every arm is scored with the camera-controlled protocol (requested occlusion = camera pan), incl. MG2 and Waypoint.",
    "Judge: local Qwen3-VL-32B (vLLM), 3-vote majority, 32 frames. The paper's judge is Gemini, so absolute values are "
    "NOT comparable to the StEvo leaderboard; compare arms only within this figure.",
    "Criteria (stevo_bench_code/eval/summarize_results.py): control success = occlusion done AND trigger applied; "
    "task success = state evolution AND NOT physical inaccuracy.",
    f"Error bars: Wilson 95{NB}% score interval, n = {N} (binomial s.e. ≈ 0.05–0.08; unpaired differences < ~0.15 are not "
    "significant). Paired p-values: exact two-sided McNemar on the tasks where exactly one of the two arms is flagged.",
    "Not run: SDVG\u00a0τ\u00a0−2.2 (disk) and ours teacher-parity mode (chunk-wise 0.7 + safe commit; time).",
    "Source: lingbot-world-v2/specwm_out/stevo38/eval_runs/lingbot_b38_<arm>/summary.json (tasks[].llm_evals); "
    "paper/meeting_1012/facts.md §14; numbers and paired tests: paper/meeting_1012/data/f12_stevo.json.",
]

# ---------------------------------------------------------------- figure layout
order = sorted(ARMS, key=lambda a: (R[a][PI]["k"], -R[a]["task_success"]["k"], ARMS.index(a)))
NROW = len(order)
FIG_W = 13.33
ROW_IN = 0.60
XL = 0.012
TITLE_KW = dict(fontsize=13.0, fontweight="bold", color=INK, linespacing=1.3)
SUB_KW = dict(fontsize=10, color=INK2, linespacing=1.35)
FOOT_KW = dict(fontsize=8.3, color=MUTED, linespacing=1.36)

fig = plt.figure(figsize=(FIG_W, 8.0), facecolor="white")
fig.canvas.draw()
renderer = fig.canvas.get_renderer()


def wrap_measured(sentences, fontsize, max_px, **kw):
    probe = fig.text(0, 0, "", fontsize=fontsize, **kw)
    lines, cur = [], ""
    for word in " ".join(sentences).split(" "):
        trial = word if not cur else cur + " " + word
        probe.set_text(trial)
        if probe.get_window_extent(renderer).width > max_px and cur:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    lines.append(cur)
    probe.remove()
    return "\n".join(ln.strip() for ln in lines)


def text_in(s, **kw):
    t = fig.text(0, 0, s, **kw)
    h = t.get_window_extent(renderer).height / fig.dpi
    t.remove()
    return h


wpx = 0.970 * FIG_W * fig.dpi
title = wrap_measured([title], TITLE_KW["fontsize"], wpx, fontweight="bold")
sub_w = "\n".join(wrap_measured([s], SUB_KW["fontsize"], wpx) for s in sub)
foot_w = wrap_measured(foot, FOOT_KW["fontsize"], wpx)
title_in = text_in(title, **TITLE_KW)
sub_in = text_in(sub_w, **SUB_KW)
foot_in = text_in(foot_w, **FOOT_KW)
HEAD_IN = 1.16  # panel group title + metric names + definitions
AXIS_IN = 0.50  # tick labels + x label
FIG_H = 0.20 + title_in + 0.16 + sub_in + 0.30 + HEAD_IN + NROW * ROW_IN + AXIS_IN + 0.22 + foot_in + 0.12
fig.set_size_inches(FIG_W, FIG_H)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
IN = 1.0 / FIG_H

fig.text(XL, 1 - 0.20 * IN, title, ha="left", va="top", **TITLE_KW)
fig.text(XL, 1 - (0.20 + title_in + 0.16) * IN, sub_w, ha="left", va="top", **SUB_KW)
fig.text(XL, 0.12 * IN, foot_w, ha="left", va="bottom", **FOOT_KW)
AX_B = (0.12 + foot_in + 0.22 + AXIS_IN) * IN
AX_H = NROW * ROW_IN * IN
AX_T = AX_B + AX_H
HEAD_TOP = AX_T + HEAD_IN * IN

A_X, A_W = 0.200, 0.255   # panel a
B_X0, B_X1 = 0.498, 0.988  # panel b small multiples
GAP = 0.020
SW = (B_X1 - B_X0 - 4 * GAP) / 5
YLIM = (NROW - 0.5, -0.5)
ypos = {a: i for i, a in enumerate(order)}
band = [ypos[a] for a in B14]
assert max(band) - min(band) == len(B14) - 1, "14B-based arms are not contiguous after sorting"
band_lo, band_hi = min(band) - 0.5, max(band) + 0.5


def style(ax, xmax, ticks):
    ax.set_facecolor("white")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_ylim(*YLIM)
    ax.set_xlim(0, xmax)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="x", length=3, colors=INK2)
    for t in ticks:
        ax.axvline(t, color=GRID, lw=0.8, zorder=0)
    ax.axhspan(band_lo, band_hi, color=BAND, lw=0, zorder=-1)


# ---- panel a: physical inaccuracy
axA = fig.add_axes([A_X, AX_B, A_W, AX_H])
XMAX_A = 1.30
style(axA, XMAX_A, [0, 0.25, 0.5, 0.75, 1.0])
axA.spines["bottom"].set_bounds(0, 1.0)
for a in order:
    r, y = R[a][PI], ypos[a]
    axA.barh(y, r["frac"], height=0.56, color=C[a], lw=0, zorder=2)
    halo = [pe.withStroke(linewidth=3.2, foreground="white")]  # keeps the whisker visible on dark bars
    axA.plot([r["lo"], r["hi"]], [y, y], color=INK2, lw=1.4, zorder=3, solid_capstyle="butt", path_effects=halo)
    for x in (r["lo"], r["hi"]):
        axA.plot([x, x], [y - 0.12, y + 0.12], color=INK2, lw=1.4, zorder=3, path_effects=halo)
    axA.text(0.02, y, f"{r['frac']:.2f} ({r['k']}/{N})", color="white", fontsize=9.2, fontweight="bold",
             ha="left", va="center", zorder=4)
    axA.text(r["hi"] + 0.02, y, f"[{r['lo']:.2f}, {r['hi']:.2f}]", color=INK2, fontsize=8.5, ha="left",
             va="center", zorder=4)
axA.text(XMAX_A - 0.005, (band_lo + min(band) + 0.5) / 2 + 0.25, "14B-based arms", color=MUTED, fontsize=8.5,
         style="italic", ha="right", va="center", zorder=4)
axA.set_xlabel("share of 38 tasks flagged (↓ lower is better)", color=INK, fontsize=9.5)

# row labels (left of panel a)
yax = axA.get_yaxis_transform()
for a in order:
    y = ypos[a]
    col = C[a] if a in MODES else INK
    axA.annotate(NAME[a], xy=(-0.03, y), xycoords=yax, xytext=(0, 6.5), textcoords="offset points", ha="right",
                 va="center", fontsize=10, fontweight="bold", color=col, annotation_clip=False)
    axA.annotate(DESC[a], xy=(-0.03, y), xycoords=yax, xytext=(0, -7.0), textcoords="offset points", ha="right",
                 va="center", fontsize=8.5, color=MUTED, annotation_clip=False)

# panel headings
HDR_Y = HEAD_TOP - 0.04 * IN
fig.text(XL, HDR_Y, "a", fontsize=13, fontweight="bold", color=INK, ha="left", va="top")
fig.text(XL + 0.018, HDR_Y, "Physical inaccuracy  ↓", fontsize=11, color=INK, ha="left", va="top")
fig.text(XL + 0.018, HDR_Y - 0.25 * IN,
         "bar = share of 38 tasks with a physical inaccuracy detected;\nwhisker + [ ] = Wilson 95 % CI (n = 38); sorted",
         fontsize=8.5, color=MUTED, ha="left", va="top", linespacing=1.3)
fig.text(B_X0 - 0.018, HDR_Y, "b", fontsize=13, fontweight="bold", color=INK, ha="left", va="top")
fig.text(B_X0, HDR_Y, "Other StEvo criteria  ↑  (same rows)", fontsize=11, color=INK, ha="left", va="top")
fig.text(B_X0, HDR_Y - 0.25 * IN,
         "dot = share of 38 tasks, line = Wilson 95 % CI; bold value = best arm in that criterion",
         fontsize=8.5, color=MUTED, ha="left", va="top")

# ---- panel b: small multiples
small = METRICS[1:]
for j, (m, _, lab, dfn, better) in enumerate(small):
    x0 = B_X0 + j * (SW + GAP)
    ax = fig.add_axes([x0, AX_B, SW, AX_H])
    style(ax, 1.0, [0, 0.5, 1.0])
    if j:
        ax.spines["left"].set_visible(False)
    tl = ax.get_xticklabels()  # keep "1" of one panel and "0" of the next apart
    tl[0].set_ha("left")
    tl[-1].set_ha("right")
    best = max(R[a][m]["k"] for a in ARMS)
    for a in order:
        r, y = R[a][m], ypos[a]
        ax.plot([r["lo"], r["hi"]], [y, y], color=lighten(C[a], 0.35), lw=2.2, zorder=2, solid_capstyle="round")
        ax.scatter([r["frac"]], [y], s=34, color=C[a], ec="white", lw=0.8, zorder=3)
        isbest = r["k"] == best and best > 0
        ax.text(r["frac"], y - 0.20, f"{r['frac']:.2f}", fontsize=8.3, ha="center", va="bottom",
                color=INK if isbest else INK2, fontweight="bold" if isbest else "normal", zorder=4,
                clip_on=False)
    ax.set_xlabel("share ↑", color=INK, fontsize=9)
    tx = x0 + SW / 2
    fig.text(tx, AX_T + 0.50 * IN, lab, fontsize=9.8, color=INK, ha="center", va="bottom")
    d = wrap_measured([dfn], 7.8, SW * FIG_W * fig.dpi * 1.02)
    fig.text(tx, AX_T + 0.46 * IN, d, fontsize=7.8, color=MUTED, ha="center", va="top", linespacing=1.2)

for ext in ("png", "svg"):
    fig.savefig(os.path.join(OUT_FIG, f"f12_stevo.{ext}"), dpi=200, facecolor="white")
print("saved", os.path.join(OUT_FIG, "f12_stevo.png"), "size", FIG_W, round(FIG_H, 2))


# ---------------------------------------------------------------- text overlap check
def check_overlaps():
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    ts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    bb = [(t, t.get_window_extent(rr)) for t in ts]
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


check_overlaps()

# ---------------------------------------------------------------- data + table
out = dict(
    source="lingbot-world-v2/specwm_out/stevo38/eval_runs/lingbot_b38_<arm>/summary.json (tasks[].llm_evals."
           + JUDGE + ")",
    protocol=f"StEvo-Bench subset, {N} camera-WM tasks, seed 1234, local Qwen3-VL-32B judge (3-vote majority, 32 frames); "
             "not comparable to the paper's Gemini leaderboard",
    ci="Wilson 95 % score interval, n = 38",
    paired_test="exact two-sided McNemar on physical_inaccuracy; only_a = tasks flagged for a but not b; uncorrected",
    not_run=["SDVG tau -2.2 (disk)", "ours teacher-parity mode (cw07_safe; time)"],
    arms={a: {m: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in R[a][m].items()} for m, *_ in METRICS}
          for a in ARMS},
    paired_physical_inaccuracy={k: dict(only_a=v["only_a"], only_b=v["only_b"], p=round(v["p"], 4))
                                for k, v in PAIRS.items()},
    order_in_figure=order,
)
json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("saved", OUT_JSON)

hdr = ["arm"] + [f"{lab} {'↓' if b == 'lo' else '↑'}" for _, _, lab, _, b in METRICS]
lines = ["# f12 StEvo-38 table (fractions of 38 tasks, Wilson 95 % CI in brackets)", "",
         f"StEvo-Bench subset: {N} camera-WM tasks, seed 1234, local Qwen3-VL-32B judge (3-vote majority, 32 frames); "
         "not comparable to the paper's Gemini leaderboard. Bold = best, underlined = second best per column "
         "(on the displayed rounding). SDVG τ −2.2 and ours teacher-parity mode not run.", "",
         "| " + " | ".join(hdr) + " |", "|---" * len(hdr) + "|"]
rank = {}
for m, _, _, _, b in METRICS:
    vals = sorted({round(R[a][m]["frac"], 2) for a in ARMS}, reverse=(b == "hi"))
    rank[m] = vals
for a in order:
    cells = []
    for m, _, _, _, b in METRICS:
        r = R[a][m]
        v = round(r["frac"], 2)
        s = f"{v:.2f}"
        if v == rank[m][0]:
            s = f"**{s}**"
        elif len(rank[m]) > 1 and v == rank[m][1]:
            s = f"<u>{s}</u>"
        cells.append(f"{s} [{r['lo']:.2f}, {r['hi']:.2f}]")
    lines.append(f"| {NAME[a].replace(NB, ' ')} | " + " | ".join(cells) + " |")
lines += ["", "Paired, physical inaccuracy (exact McNemar, two-sided, uncorrected; a-only / b-only flagged tasks, p):"]
for k, v in PAIRS.items():
    a, b = k.split("_vs_")
    lines.append(f"- {SHORT[a].replace(NB, ' ')} vs {SHORT[b].replace(NB, ' ')}: {v['only_a']} / {v['only_b']}, "
                 f"{fmt_p(v['p']).replace(NB, ' ')}")
lines += ["", "Generated by paper/meeting_1012/scripts/f12_stevo.py from lingbot-world-v2/specwm_out/stevo38/eval_runs/"
          "lingbot_b38_<arm>/summary.json."]
open(OUT_TAB, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("saved", OUT_TAB)
for a in order:
    print(a, {m: f"{R[a][m]['k']}/{N}" for m, *_ in METRICS})
for k, v in PAIRS.items():
    print(k, v)
