"""f02_partial_commit: why fused commit jumps and how partial commit fixes it.

(a) schematic of the K/V written for one chunk (3 latent frames) under separate / fused / partial commit;
(b) share of chunk boundaries with a jump (r > 1.5) per arm, parsed from facts.md section 5.
Timings: facts.md section 15 quiet timing pass wherever it has the same arm (teacher, 704 separate / fused /
fused + partial 1, 832 fused + partial 1), marked *; the other rows keep their section-5 timing. The partial-commit
cost in the title is the same-condition difference 0.763 - 0.630 s/chunk (section 15), NOT 0.761 - 0.640 (section 5,
under load vs quiet). Mid-chunk control band = min-max of the section-5 control column (0.7-3.7 %).
Re-run:  CUDA_VISIBLE_DEVICES="" /home/hk28965/specwm/venv/bin/python paper/meeting_1012/scripts/f02_partial_commit.py
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.transforms import blended_transform_factory

ROOT = "/home/hk28965/specwm"
MEET = os.path.join(ROOT, "paper/meeting_1012")
FACTS = os.path.join(MEET, "facts.md")
OUT = os.path.join(MEET, "figs", "f02_partial_commit")

# ---------------------------------------------------------------- style
INK, INK2, MUTED = "#1f1f1f", "#4d4d4d", "#8a8a8a"
GRID = "#e6e6e6"
C_TEACHER, C_OURS832, C_OURS704 = "#1f1f1f", "#7b4bd1", "#d81b60"
C_GREY = "#9e9e9e"
C_CLEAN, C_NOISY_FACE, C_NOISY_EDGE = "#546e7a", "#f1f3f4", "#90a4ae"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.edgecolor": "#bdbdbd", "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK, "hatch.linewidth": 0.8, "svg.fonttype": "none",
})


# ---------------------------------------------------------------- data (facts.md section 5)
def load_section5(path):
    txt = open(path, encoding="utf-8").read()
    m = re.search(r"\n## 5\..*?\n(.*?)(?=\n## )", txt, re.S)
    assert m, "facts.md: section 5 not found"
    rows = [l for l in m.group(1).splitlines() if l.startswith("|")]
    head = [c.strip() for c in rows[0].strip().strip("|").split("|")]
    out = {}
    for l in rows[2:]:
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        out[cells[0]] = dict(zip(head, cells))
    return head, out


HEAD, S5 = load_section5(FACTS)


def load_section15_row(path, key):
    """(steady s/chunk, fps) of one row of the facts.md section 15 quiet timing-pass table"""
    txt = open(path, encoding="utf-8").read()
    m = re.search(r"\n## 15\..*?\n(.*?)(?=\n## |\Z)", txt, re.S)
    assert m, "facts.md: section 15 not found"
    for l in m.group(1).splitlines():
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        if l.startswith("|") and cells[0] == key:
            return cells[2], cells[3]
    raise KeyError(key)


# Teacher timing = quiet STEADY s/chunk (facts.md section 15 timing pass, jar + castle). The section-5 cell
# "2.45 s / 4.9 fps" is a whole-clip average (31.2 s / 13 chunks incl. the cheap early chunks), not steady.
TEA_STEADY_S, TEA_STEADY_FPS = load_section15_row(FACTS, "teacher 14B 4-step (uniform4)")
COL_S = next(h for h in HEAD if h.startswith("cloud s/chunk"))
COL_FPS = next(h for h in HEAD if h.startswith("fps"))
COL_J = next(h for h in HEAD if h.startswith("boundaries"))
COL_CTRL = next(h for h in HEAD if h.startswith("mid-chunk control"))

# Timing shown in this figure = the facts.md section 15 QUIET timing pass wherever it has the same arm (same
# condition for every such row). The section-5 timings mix load conditions (704 fused 0.640 s quiet vs
# 704 fused + partial 1 0.761 s under load), so their 0.12 s difference is NOT the partial-commit cost (facts §1).
S15 = {  # facts §5 row key -> facts §15 row key
    "teacher 832 (s1234)": "teacher 14B 4-step (uniform4)",
    "832 fused + partial 1 (s1234)": "ours quality mode (832 fused + partial)",
    "704 separate": "704, separate commit",
    "704 fused": "704 fused (jumps)",
    "704 fused + partial 1": "ours realtime mode (704 fused + partial)",
}
Q15 = {k: load_section15_row(FACTS, v) for k, v in S15.items()}
FUSED_Q, PARTIAL_Q = float(Q15["704 fused"][0]), float(Q15["704 fused + partial 1"][0])
D_PARTIAL = PARTIAL_Q - FUSED_Q  # 0.763 - 0.630 = +0.133 s/chunk, same condition (quiet)

# same-task check from the raw files facts §15 names (cloud_chunk_s[6:], per-clip median, median over the
# clips of one task; seed 1 = warm-up and castle s1234 = cold compile excluded, as in §15)
TP = os.path.join(ROOT, "lingbot-world-v2/specwm_out/timing_pass")


def _task_median(task, arm):
    import glob
    import json
    import statistics as st
    v = []
    for f in sorted(glob.glob(os.path.join(TP, task, arm + "_s*", "loop.json"))):
        seed = f.split(os.sep)[-2].rsplit("_s", 1)[1]
        if seed == "1" or (task.startswith("sandcastle") and seed == "1234"):
            continue
        c = json.load(open(f))["cloud_chunk_s"][6:]
        v.append(st.median(c))
    return st.median(v) if v else None


SAME_TASK = {t: (_task_median(t, "c704_fused"), _task_median(t, "ours704"))
             for t in ("sand_filling_up_jar", "sandcastle_wave_impact")}
HAVE_RAW = all(a is not None and b is not None for a, b in SAME_TASK.values())


def control_range():
    """min / max of the mid-chunk control share (%) over the nine §5 arms"""
    vals = [float(re.search(r"\(([\d.]+) %\)", S5[k][COL_CTRL]).group(1)) for k in S5]
    return min(vals), max(vals)


CTRL_LO, CTRL_HI = control_range()  # 0.7 / 3.7 % (facts §1, §5)

# (facts.md row key, display label, colour, fused without partial -> hatched)
ARMS = [
    ("teacher 832 (s1234)", "Teacher 14B 4-step 832, separate  (s1234)", C_TEACHER, False),
    ("832 fused (s1234)", "832 2-step fused  (s1234)", C_GREY, True),
    ("832 fused + partial 1 (s1234)", "832 fused + partial 1 = ours 832  (s1234)", C_OURS832, False),
    ("704 separate", "704 2-step separate", C_GREY, False),
    ("704 fused", "704 2-step fused", C_GREY, True),
    ("704 fused + partial 1", "704 fused + partial 1 = ours 704", C_OURS704, False),
    ("704 fused + partial 2", "704 fused + partial 2", C_GREY, False),
    ("672 fused", "672 2-step fused", C_GREY, True),
    ("576 fused", "576 2-step fused", C_GREY, True),
]
for k, *_ in ARMS:
    assert k in S5, "facts.md section 5 has no row %r" % k


def pct(s):
    return float(s.replace("%", "").strip())


def sec(key):
    return S5[key][COL_S]


def sec_q(key):
    """quiet steady s/chunk (facts §15) for the panel-a rows"""
    return Q15[key][0]


fig = plt.figure(figsize=(12, 6.75), facecolor="white")
j704f, j704p = pct(S5["704 fused"][COL_J]), pct(S5["704 fused + partial 1"][COL_J])
TITLE = ("Partial commit: fused-commit jumps %.0f %% → %.0f %% of chunk boundaries at 704, "
         "for +%.2f s per chunk (quiet)" % (j704f, j704p, D_PARTIAL))
ttl = fig.text(0.015, 0.965, TITLE, fontsize=13, fontweight="bold", color=INK, va="top")

# ================================================================ (a) schematic
axA = fig.add_axes([0.012, 0.15, 0.44, 0.74])
axA.set_xlim(0, 10)
axA.set_ylim(0, 10)
axA.axis("off")
fig.text(0.015, 0.895, "(a) What one chunk leaves in the KV cache", fontsize=11, fontweight="bold", color=INK, va="bottom")

CX = [3.25, 4.5, 5.75]     # left x of the f1/f2/f3 cells
CW, CH = 1.12, 1.05
XF, FW = 7.35, 1.75        # extra-commit column: x, width of one full 14B forward
ROWS = [  # name, sub-label, cell states (True = clean), forwards, cost text, facts key
    ("Separate", "extra clean t=0 pass\nover f1–f3", (True, True, True), 1.0, "+1 forward", "704 separate"),
    ("Fused", "keep the K/V of the\nlast denoising forward", (False, False, False), 0.0, "+0 forward", "704 fused"),
    ("Partial (ours)", "fused, then a clean t=0\npass on f3 tokens only", (False, False, True), 1 / 3, "+⅓ forward",
     "704 fused + partial 1"),
]
YR = [5.75, 3.85, 1.95]    # bottom y of each row's cells

# column headers
for x, lab in zip(CX, ["f1", "f2", "f3 (newest)"]):
    axA.text(x + CW / 2, 7.2, lab, ha="center", va="bottom", fontsize=10, color=INK)
axA.text((CX[0] + CX[2] + CW) / 2, 7.75, "latent frames of the chunk (oldest → newest)", ha="center", va="bottom",
         fontsize=9, color=MUTED)
axA.text(XF, 7.2, "extra commit work\nper chunk, 704 (quiet)", ha="left", va="bottom", fontsize=9.5, color=INK2,
         linespacing=1.05)

# next chunk box + attention arrows (thick = newest frame)
bx, by, bw, bh = 7.6, 8.85, 2.0, 0.6
axA.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white", ec=INK2, lw=1.2))
axA.text(bx + bw / 2, by + bh / 2, "next chunk", ha="center", va="center", fontsize=10, color=INK)
for i, rad in enumerate([0.28, 0.26, 0.22]):
    newest = i == 2
    axA.add_patch(FancyArrowPatch((bx, by + bh / 2), (CX[i] + CW / 2, 8.3 if not newest else 8.25),
                                  connectionstyle="arc3,rad=%.2f" % rad,
                                  arrowstyle="-|>,head_length=9,head_width=5.5" if newest else "-|>,head_length=6,head_width=3.5",
                                  lw=3.0 if newest else 0.9, color=INK if newest else MUTED,
                                  linestyle="-" if newest else (0, (4, 3)), shrinkA=0, shrinkB=0))
axA.text(0.0, 9.4, "next chunk reads the cache,\nleaning on the newest frame f3", fontsize=9.5, color=INK2, va="center",
         linespacing=1.1)
axA.text(0.0, 8.62, "(re-committing f2 too adds\nnothing: %s vs %s jumps)" % (S5["704 fused + partial 2"][COL_J],
         S5["704 fused + partial 1"][COL_J]), fontsize=9, color=MUTED, va="center", linespacing=1.1)

for (name, sub, states, fwd, ftxt, key), y in zip(ROWS, YR):
    axA.text(0.0, y + CH, name, fontsize=11, fontweight="bold", color=INK, va="top")
    axA.text(0.0, y + CH - 0.45, sub, fontsize=9.5, color=INK2, va="top", linespacing=1.1)
    for x, clean in zip(CX, states):
        if clean:
            axA.add_patch(Rectangle((x, y), CW, CH, fc=C_CLEAN, ec=C_CLEAN, lw=1.0))
            axA.text(x + CW / 2, y + CH / 2, "clean\nt = 0", ha="center", va="center", fontsize=9, color="white",
                     linespacing=1.0)
        else:
            axA.add_patch(Rectangle((x, y), CW, CH, fc=C_NOISY_FACE, ec=C_NOISY_EDGE, lw=1.0, hatch="////"))
            axA.text(x + CW / 2, y + CH / 2, "noisy\nσ 0.833", ha="center", va="center", fontsize=9, color=INK,
                     linespacing=1.0, bbox=dict(fc=C_NOISY_FACE, ec="none", pad=0.6))
    # mini bar: outline = one full 14B forward, fill = extra commit forward
    yb = y + 0.42
    axA.text(XF, yb + 0.38, ftxt, ha="left", va="bottom", fontsize=9.5, color=INK, fontweight="bold")
    axA.add_patch(Rectangle((XF, yb), FW, 0.3, fc="none", ec="#bdbdbd", lw=0.9))
    if fwd > 0:
        axA.add_patch(Rectangle((XF, yb), FW * fwd, 0.3, fc=C_CLEAN, ec=C_CLEAN, lw=0.9))
    axA.text(XF, yb - 0.1, "%s s/chunk\n%s jumps" % (sec_q(key), S5[key][COL_J]), ha="left", va="top", fontsize=9.5,
             color=INK2, linespacing=1.1)

# legend
ly = 0.75
axA.add_patch(Rectangle((0.0, ly), 0.42, 0.34, fc=C_CLEAN, ec=C_CLEAN))
axA.text(0.55, ly + 0.17, "clean K/V: the denoised x0 re-encoded at t = 0", fontsize=9.5, color=INK, va="center")
axA.add_patch(Rectangle((0.0, ly - 0.6), 0.42, 0.34, fc=C_NOISY_FACE, ec=C_NOISY_EDGE, hatch="////"))
axA.text(0.55, ly - 0.43, "noisy K/V: written by the last denoising forward (input at σ 0.833)", fontsize=9.5,
         color=INK, va="center")

# ================================================================ (b) evidence
# (row kind, facts key or header text, display label, colour, hatched = fused without partial)
LAYOUT = [
    ("h", "832  (seed 1234 only)"),
    ("a", "teacher 832 (s1234)", "teacher 14B, 4-step", C_TEACHER, False),
    ("a", "832 fused (s1234)", "2-step fused", C_GREY, True),
    ("a", "832 fused + partial 1 (s1234)", "fused + partial 1 (ours)", C_OURS832, False),
    ("h", "704  (3 seeds)"),
    ("a", "704 separate", "2-step separate", C_GREY, False),
    ("a", "704 fused", "2-step fused", C_GREY, True),
    ("a", "704 fused + partial 1", "fused + partial 1 (ours)", C_OURS704, False),
    ("a", "704 fused + partial 2", "fused + partial 2", C_GREY, False),
    ("h", "lower resolution  (3 seeds)"),
    ("a", "672 fused", "672, 2-step fused", C_GREY, True),
    ("a", "576 fused", "576, 2-step fused", C_GREY, True),
]
assert sorted(r[1] for r in LAYOUT if r[0] == "a") == sorted(k for k, *_ in ARMS)
axB = fig.add_axes([0.66, 0.255, 0.175, 0.605])
n = len(LAYOUT)
xmax = 22
axB.axvspan(CTRL_LO, CTRL_HI, color="#efefef", zorder=0, lw=0)
yt, yl = [], []
hdr_tf = blended_transform_factory(fig.transFigure, axB.transData)
for i, row in enumerate(LAYOUT):
    y = n - 1 - i
    if row[0] == "h":
        axB.text(0.478, y - 0.05, row[1], transform=hdr_tf, ha="left", va="center", fontsize=10, fontweight="bold",
                 color=INK2)
        if i > 0:
            axB.axhline(y + 0.5, color="#d0d0d0", lw=0.8, ls=(0, (3, 3)), zorder=1, xmin=-0.0)
        continue
    _, k, lab, col, hatched = row
    v = pct(S5[k][COL_J])
    if v > 0:
        axB.barh(y, v, height=0.64, color="#e3e3e3" if hatched else col, edgecolor="#757575" if hatched else col,
                 hatch="////" if hatched else None, lw=0.9, zorder=3)
    else:
        axB.plot([0.12, 0.12], [y - 0.32, y + 0.32], color=col, lw=3.5, zorder=3, solid_capstyle="butt")
    axB.text(v + 0.45, y, "%g %%" % v, va="center", ha="left", fontsize=10, color=INK, zorder=4,
             fontweight="bold" if col in (C_OURS704, C_OURS832) else "normal")
    yt.append(y)
    yl.append(lab)
    s_txt, f_txt, mark = sec(k), S5[k][COL_FPS], ""
    if k in Q15:  # quiet steady value of the same arm (facts §15); teacher: not the §5 whole-clip average
        s_txt, f_txt, mark = Q15[k][0], Q15[k][1], " *"
        if k == "teacher 832 (s1234)":
            s_txt = "%.2f" % float(TEA_STEADY_S)
    axB.text(1.08, y, "%s s · %s fps%s" % (s_txt, f_txt, mark), transform=axB.get_yaxis_transform(),
             ha="left", va="center", fontsize=9.5, color=INK2)
axB.set_yticks(yt)
axB.set_yticklabels(yl)
for t in axB.get_yticklabels():
    if "(ours)" in t.get_text():
        t.set_fontweight("bold")
axB.set_xlim(0, xmax)
axB.set_ylim(-0.6, n - 0.4)
axB.set_xlabel("chunk boundaries with a jump, r > 1.5 (%)\n↓ lower is better", fontsize=11)
axB.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
axB.set_axisbelow(True)
for s_ in ("top", "right"):
    axB.spines[s_].set_visible(False)
axB.tick_params(axis="y", length=0)
axB.text(1.08, n - 1, "cloud s/chunk ↓\n· fps ↑", transform=axB.get_yaxis_transform(), ha="left",
         va="center", fontsize=9.5, color=INK2, linespacing=1.05)
# key: hatched swatch in the 832 header row, band label in the 704 header row (no bars there)
kx, ky = 9.0, n - 1
axB.add_patch(Rectangle((kx, ky - 0.25), 1.2, 0.45, fc="#e3e3e3", ec="#757575", hatch="////", lw=0.9, zorder=3))
axB.text(kx + 1.5, ky - 0.03, "fused, no partial", va="center", fontsize=9, color=INK)
y704 = n - 1 - [r[1] for r in LAYOUT].index("704  (3 seeds)")
axB.text(CTRL_HI + 0.4, y704 - 0.03, "← mid-chunk control: %g–%g %%" % (CTRL_LO, CTRL_HI), va="center", fontsize=9,
         color=INK2)
fig.text(0.478, 0.895, "(b) Chunk-boundary jumps per arm (12 prompts)", fontsize=11, fontweight="bold", color=INK,
         va="bottom", ha="left")

# ---------------------------------------------------------------- footer
jar_f, jar_p = SAME_TASK["sand_filling_up_jar"]
cas_f, cas_p = SAME_TASK["sandcastle_wave_impact"]
FOOT = [
    "Data: facts.md §5 (= paper/1012/README.md §W): 12 prompts × 3 seeds = 36 clips per arm (832 rows: seed 1234 "
    "only, 12 clips). Metric: scripts/boundary_jump.py,",
    "r = consecutive-frame mean abs diff (grey, 208 px) at a chunk boundary ÷ mean of its 4 neighbouring steps; "
    "bars = share of boundaries with r > 1.5; grey band =",
    "mid-chunk control (same r at b+4 / b+8): %g–%g %% across the nine arms. * = quiet steady timing pass "
    "(facts §15: jar + sandcastle, generation only, one RTX PRO" % (CTRL_LO, CTRL_HI),
    "6000 Blackwell); the teacher's older “2.45 s · 4.9 fps quiet” was a whole-clip average. Unmarked timings: "
    "facts §5 (672 fused quiet; 832 fused, partial 2, 576 fused: load n/a).",
    ("Partial cost +%.2f s = %s − %s s/chunk (§15 medians over 5 vs 6 clips); same task and seeds (raw §15 "
     "loop.json): jar %.3f → %.3f, sandcastle %.3f → %.3f (+%.2f / +%.2f)."
     % (D_PARTIAL, Q15["704 fused + partial 1"][0], Q15["704 fused"][0], jar_f, jar_p, cas_f, cas_p,
        jar_p - jar_f, cas_p - cas_f)) if HAVE_RAW else
    ("Partial cost +%.2f s = %s − %s s/chunk (§15 medians over 5 vs 6 clips: castle s1234 excluded for ours 704)."
     % (D_PARTIAL, Q15["704 fused + partial 1"][0], Q15["704 fused"][0])),
]
for i, line in enumerate(FOOT):
    fig.text(0.015, 0.012 + 0.026 * (len(FOOT) - 1 - i), line, fontsize=9, color=MUTED, va="bottom")

# layout check: every text inside the canvas, no two texts overlapping
fig.canvas.draw()
_r = fig.canvas.get_renderer()
_W, _H = fig.bbox.width, fig.bbox.height
_ts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
_bb = [(t, t.get_window_extent(_r)) for t in _ts]
_bad = 0
for _i, (_t, _b) in enumerate(_bb):
    if _b.x0 < 4 or _b.y0 < 2 or _b.x1 > _W - 4 or _b.y1 > _H - 2:
        print("WARN outside canvas:", repr(_t.get_text()[:60]), round(_b.x1 - _W))
        _bad += 1
    for _t2, _b2 in _bb[_i + 1:]:
        ox = min(_b.x1, _b2.x1) - max(_b.x0, _b2.x0)
        oy = min(_b.y1, _b2.y1) - max(_b.y0, _b2.y0)
        if ox > 1 and oy > 1:
            print("WARN text overlap:", repr(_t.get_text()[:40]), "<>", repr(_t2.get_text()[:40]))
            _bad += 1
print("layout check:", "OK" if not _bad else "%d problems" % _bad)
print("title:", TITLE)
print("partial cost (quiet, §15):", round(D_PARTIAL, 4), "same task:", SAME_TASK, "control range:", CTRL_LO, CTRL_HI)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT + ".png", dpi=200, facecolor="white")
fig.savefig(OUT + ".svg", facecolor="white")
print("wrote", OUT + ".png", OUT + ".svg")
