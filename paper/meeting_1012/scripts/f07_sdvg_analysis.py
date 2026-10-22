#!/usr/bin/env python
"""f07_sdvg_analysis: SDVG time breakdown (left) and quality vs generate time (right).

Re-run:  CUDA_VISIBLE_DEVICES="" /home/hk28965/specwm/venv/bin/python \
         /home/hk28965/specwm/paper/meeting_1012/scripts/f07_sdvg_analysis.py

Data sources (paths relative to /home/hk28965/specwm):
  - paper/meeting_1012/facts.md  (canonical numbers: generate totals, decode, prepare,
    SDVG component medians, accept-all ceiling and its "x the student" ratio)
  - paper/1005/meeting/05_sdvg_breakdown/breakdown.json  (10/5 original-code breakdown,
    incl. chunks accepted from the draft)
  - lingbot-world-v2/specwm_out/bench43/*/<arm>_s{1234,4321}/loop.json  (per-clip times;
    used for the teacher/student denoise/commit split and the student prepare time,
    which facts.md does not list, to cross-check every facts number, and for chunks per
    clip, the forced first chunk and max_area)
  - lingbot-world-v2/specwm_out/bench43/*/<arm>_s*/video.mp4  (frame count, fps, frame size
    for the footer; read with OpenCV, header only)
  - paper/1012/bench43_summary.json  (VLM quality mean +- SE, chunks taken from draft)
No GPU is used.

Drawing rules: each bar = the generate components (draft, verify, target, commit), a
black tick where generate ends with the median generate seconds in bold above it, then
the final VAE decode as a thinner light segment. Prepare (T5 + image encode, runs before
generate) is not drawn because it was not recorded in the optimized SDVG runs; it is
given in the row labels instead.
"""
import collections
import glob
import json
import os
import re
import textwrap

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

ROOT = "/home/hk28965/specwm"
OUT = os.path.join(ROOT, "paper/meeting_1012/figs")
FACTS = os.path.join(ROOT, "paper/meeting_1012/facts.md")
BREAKDOWN = os.path.join(ROOT, "paper/1005/meeting/05_sdvg_breakdown/breakdown.json")
BENCH = os.path.join(ROOT, "lingbot-world-v2/specwm_out/bench43")
SUMMARY = os.path.join(ROOT, "paper/1012/bench43_summary.json")

# ---------------------------------------------------------------- style
ARM = {"teacher": "#1f1f1f", "student": "#e07b00", "sdvg07": "#8d6e63",
       "sdvg22": "#c49a6c", "other": "#9e9e9e"}
COMP = {  # component colours (draft/target reuse the student/teacher model colour)
    "draft": dict(fc="#e07b00", hatch=None, txt="white"),
    "verify": dict(fc="#3f7fd6", hatch=None, txt="white"),
    "target": dict(fc="#1f1f1f", hatch=None, txt="white"),
    "commit": dict(fc="#8a8a8a", hatch=None, txt="white"),
    "decode": dict(fc="#dfe7f5", hatch="////", txt="#1f1f1f", hc="#9fb2d3"),
}
COMP_LABEL = {
    "draft": "1.3B denoise (SDVG draft / student)",
    "verify": "verify: VAE decode + ImageReward",
    "target": "14B denoise (SDVG target / teacher)",
    "commit": "KV commit (clean t=0 forward; SDVG: both caches)",
    "decode": "final VAE decode (after generate; thinner bar)",
}
INK, INK2, MUTED, GRID = "#1f1f1f", "#4a4a4a", "#7a7a7a", "#e6e6e6"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 13,
    "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.edgecolor": "#9a9a9a", "axes.linewidth": 0.8, "hatch.linewidth": 0.8,
    "svg.fonttype": "none",
})


def style_axes(ax, grid_axis):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, length=3)


# ---------------------------------------------------------------- facts.md
facts = open(FACTS, encoding="utf-8").read()
num = r"([0-9]+(?:\.[0-9]+)?)"
gen_line = next(l for l in facts.splitlines() if l.startswith("Generation seconds per 10 s clip"))
m = re.search(r"teacher %s \(\+ decode %s full VAE, \+ prepare %s\)" % (num, num, num), gen_line)
F = {"teacher": dict(gen=float(m[1]), decode=float(m[2]), prepare=float(m[3]))}
m = re.search(r"student %s \(\+ decode %s\)" % (num, num), gen_line)
F["student"] = dict(gen=float(m[1]), decode=float(m[2]))
m = re.search(r"SDVG−0\.7 %s \(draft %s, verify %s, target %s, commit %s; \+ final decode %s\)"
              % ((num,) * 6), gen_line)
F["sdvg07"] = dict(gen=float(m[1]), draft=float(m[2]), verify=float(m[3]), target=float(m[4]),
                   commit=float(m[5]), decode=float(m[6]))
m = re.search(r"SDVG−2\.2 %s \(draft %s, verify %s, target %s, commit %s\)" % ((num,) * 5), gen_line)
F["sdvg22"] = dict(gen=float(m[1]), draft=float(m[2]), verify=float(m[3]), target=float(m[4]),
                   commit=float(m[5]))
m = re.search(r"SDVG accept-all ceiling.*?draft %s \+ verify %s \+ commit to both caches %s "
              r"≈ %s s/chunk ≈ %s s/clip.*?still %s× slower than the student" % ((num,) * 6), facts)
CEIL = dict(draft=m[1], verify=m[2], commit=m[3], per_chunk=m[4], clip=float(m[5]), x_student=m[6])
m = re.search(r"10/5 slide .*?total %s s;" % num, facts)
ORIG_TOTAL_FACTS = float(m[1])

# ---------------------------------------------------------------- raw: 10/5 breakdown
bd = json.load(open(BREAKDOWN, encoding="utf-8"))
orig = next(v for k, v in bd.items() if k.startswith("SDVG") and "0.7" in k)
ORIG = dict(prepare=orig["prepare"], draft=orig["draft"],
            verify=orig["vae_verify"] + orig["imagereward"], target=orig["target"],
            commit=orig["commit"], decode=orig["final_decode"], gen=orig["gen"], e2e=orig["e2e"],
            accepted=orig["accepted"])
assert abs(ORIG["e2e"] - ORIG_TOTAL_FACTS) < 0.06, (ORIG["e2e"], ORIG_TOTAL_FACTS)

# ---------------------------------------------------------------- raw: bench-43 loop.json
RAW_ARM = {"teacher": "uniform4", "student": "uniform4_1p3b", "sdvg07": "ir_-0.7", "sdvg22": "ir_-2.2"}


def load_rows(arm):
    rows = []
    for s in (1234, 4321):
        for f in sorted(glob.glob(os.path.join(BENCH, "*", f"{arm}_s{s}", "loop.json"))):
            rows.append(json.load(open(f)))
    return rows


def medians(rows):
    out = {"n": len(rows)}
    for k in ("gen_s", "decode_s", "prepare_s"):
        v = [r[k] for r in rows if r.get(k) is not None]
        if v:
            out[k[:-2]] = float(np.median(v))
    for k in ("draft", "verify", "target", "commit", "denoise"):
        v = [r["times"][k] for r in rows if k in r.get("times", {})]
        if v:
            out[k] = float(np.median(v))
    return out


LOOPS = {a: load_rows(r) for a, r in RAW_ARM.items()}
R = {a: medians(rows) for a, rows in LOOPS.items()}
for a in R:
    assert R[a]["n"] == 86, (a, R[a]["n"])
    for k, v in F[a].items():  # every facts number must match the raw median to rounding
        assert abs(R[a][k] - v) < 0.051, (a, k, R[a][k], v)

# chunks per clip; SDVG forces chunk 0 to the 14B target, so at most N-1 come from the draft
N_CHUNKS = {r["chunks"] for rows in LOOPS.values() for r in rows}
assert len(N_CHUNKS) == 1, N_CHUNKS
N_CHUNKS = N_CHUNKS.pop()
sdvg_rows = LOOPS["sdvg07"] + LOOPS["sdvg22"]
assert all(r["first"] == "target" for r in sdvg_rows)
MAX_FROM_DRAFT = N_CHUNKS - 1
assert max(r["accepted"] for r in sdvg_rows) <= MAX_FROM_DRAFT
MAX_AREA = {r["max_area"] for r in sdvg_rows}
assert len(MAX_AREA) == 1, MAX_AREA
MAX_AREA = MAX_AREA.pop()

# ---------------------------------------------------------------- raw: videos (footer)
n_frames, v_fps, size_by_prompt = collections.Counter(), collections.Counter(), {}
for arm in RAW_ARM.values():
    for f in glob.glob(os.path.join(BENCH, "*", f"{arm}_s*", "video.mp4")):
        cap = cv2.VideoCapture(f)
        n_frames[int(cap.get(cv2.CAP_PROP_FRAME_COUNT))] += 1
        v_fps[cap.get(cv2.CAP_PROP_FPS)] += 1
        size_by_prompt.setdefault(f.split(os.sep)[-3], set()).add(
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        cap.release()
assert len(n_frames) == 1 and len(v_fps) == 1, (n_frames, v_fps)
FRAMES, VFPS = next(iter(n_frames)), next(iter(v_fps))
assert all(len(s) == 1 for s in size_by_prompt.values())
size_count = collections.Counter(next(iter(s)) for s in size_by_prompt.values())
(TOP_W, TOP_H), TOP_N = size_count.most_common(1)[0]
N_PROMPTS = len(size_by_prompt)

summ = json.load(open(SUMMARY, encoding="utf-8"))


# ---------------------------------------------------------------- rows for the left panel
def pick(a, k):  # facts number where facts gives one, else raw median
    return F[a].get(k, R[a].get(k))


rows = [
    dict(key="orig", colour=ARM["sdvg07"], name="SDVG τ −0.7, original", gen=ORIG["gen"], star=False,
         sub=[f"{ORIG['accepted']:.1f}/{N_CHUNKS} chunks from draft",
              f"prepare {ORIG['prepare']:.1f} s · total {ORIG_TOTAL_FACTS:.1f} s"],
         comps=dict(draft=ORIG["draft"], verify=ORIG["verify"], target=ORIG["target"],
                    commit=ORIG["commit"], decode=ORIG["decode"])),
    dict(key="sdvg07", colour=ARM["sdvg07"], name="SDVG τ −0.7, optimized", gen=F["sdvg07"]["gen"],
         star=True,
         sub=[f"{summ['sdvg07']['accepted']:.2f}/{N_CHUNKS} chunks from draft", "prepare not recorded"],
         comps=dict(draft=pick("sdvg07", "draft"), verify=pick("sdvg07", "verify"),
                    target=pick("sdvg07", "target"), commit=pick("sdvg07", "commit"),
                    decode=pick("sdvg07", "decode"))),
    dict(key="sdvg22", colour=ARM["sdvg22"], name="SDVG τ −2.2, optimized", gen=F["sdvg22"]["gen"],
         star=True,
         sub=[f"{summ['sdvg22']['accepted']:.1f}/{N_CHUNKS} chunks from draft", "prepare not recorded"],
         comps=dict(draft=pick("sdvg22", "draft"), verify=pick("sdvg22", "verify"),
                    target=pick("sdvg22", "target"), commit=pick("sdvg22", "commit"),
                    decode=R["sdvg22"]["decode"])),
    dict(key="teacher", colour=ARM["teacher"], name="teacher 14B 4-step", gen=F["teacher"]["gen"],
         star=False, sub=[f"prepare {F['teacher']['prepare']:.1f} s"],
         comps=dict(target=R["teacher"]["denoise"], commit=R["teacher"]["commit"],
                    decode=F["teacher"]["decode"])),
    dict(key="student", colour=ARM["student"], name="student 1.3B 4-step", gen=F["student"]["gen"],
         star=False, sub=[f"prepare {R['student']['prepare']:.1f} s"],
         comps=dict(draft=R["student"]["denoise"], commit=R["student"]["commit"],
                    decode=F["student"]["decode"])),
]
ORDER = ["draft", "verify", "target", "commit", "decode"]
GEN_PARTS = ("draft", "verify", "target", "commit")

# ---------------------------------------------------------------- figure + layout
FIG_W, FIG_H = 13.33, 7.5
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
fig.canvas.draw()
renderer = fig.canvas.get_renderer()


def fig_text_size(s, **kw):
    t = fig.text(0, 0, s, **kw)
    bb = t.get_window_extent(renderer=renderer)
    t.remove()
    return bb.width / fig.bbox.width, bb.height / fig.bbox.height


def wrap_to(s, width_frac, **kw):
    """Greedy word wrap to a width in figure fraction; keeps explicit newlines and never
    breaks at a non-breaking space (used to keep numbers with their units)."""
    lines = []
    for para in s.split("\n"):
        cur = ""
        for w in para.split(" "):
            trial = (cur + " " + w).strip()
            if cur and fig_text_size(trial, **kw)[0] > width_frac:
                lines.append(cur)
                cur = w
            else:
                cur = trial
        lines.append(cur)
    return "\n".join(lines)


TITLE_KW = dict(fontsize=13, linespacing=1.3)
NB = "\u00a0"
d_plus_v = F["sdvg07"]["draft"] + F["sdvg07"]["verify"]
title_L = wrap_to(
    f"Even optimized, SDVG{NB}τ{NB}−0.7 needs {F['sdvg07']['gen']:.1f}{NB}s per clip, more than the "
    f"teacher alone ({F['teacher']['gen']:.1f}{NB}s):\nonly {summ['sdvg07']['accepted']:.2f}/{N_CHUNKS} "
    f"chunks come from the draft, so the 14B still runs most chunks,\nand draft{NB}+{NB}verify add "
    f"{d_plus_v:.1f}{NB}s on top", 0.625, **TITLE_KW)
title_R = wrap_to(
    f"Even accepting every draft chunk costs ≈{NB}{CEIL['clip']:.0f}{NB}s per clip ({CEIL['x_student']}× "
    f"the student), so no threshold reaches the student's {F['student']['gen']:.1f}{NB}s; only "
    f"τ{NB}−0.7 keeps teacher quality ({summ['sdvg07']['quality']:.2f})", 0.335, **TITLE_KW)
title_h = max(fig_text_size(title_L, **TITLE_KW)[1], fig_text_size(title_R, **TITLE_KW)[1])

s07 = sum(rows[1]["comps"][c] for c in GEN_PARTS)
s22 = sum(rows[2]["comps"][c] for c in GEN_PARTS)
foot = (
    f"Bench-43: 38 StEvo camera-WM tasks + 5 easy scenes ({N_PROMPTS} prompts) × seeds 1234/4321 = 86 clips per arm; {N_CHUNKS} chunks per "
    f"clip ({FRAMES} frames ≈ {FRAMES / VFPS:.1f} s at {VFPS:.0f} fps); max_area {MAX_AREA}, frame size follows "
    f"the start image ({TOP_W}×{TOP_H} for {TOP_N} of {N_PROMPTS} prompts). Times: median seconds per clip "
    "from loop.json, one RTX PRO 6000, measured under load (7 concurrent GPU jobs, ≈10–20 % slower). "
    "Quality: Qwen3-VL-32B judge, 8 frames at 448 px, mean ± SE (quality prompt not human-validated). "
    f"Top bar: 10/5 run, original code, jar/castle{NB}×{NB}3{NB}seeds, load not recorded "
    "(paper/1005/meeting/05_sdvg_breakdown). Prepare (T5 + image encode, before generate) is not drawn: "
    "it was not recorded in the optimized SDVG runs. "
    f"* Medians do not add: the optimized SDVG components sum to {s07:.1f}{NB}/{NB}{s22:.1f}{NB}s (where "
    f"the tick sits) vs median generate {F['sdvg07']['gen']:.1f}{NB}/{NB}{F['sdvg22']['gen']:.1f}{NB}s. "
    "Teacher/student denoise vs commit split and student prepare are loop.json medians. Accept-all "
    "ceiling: estimate from the τ −2.2 per-chunk costs; only τ −0.7 and τ −2.2 were run."
)
FOOT_KW = dict(fontsize=9, linespacing=1.35)
foot = "\n".join(textwrap.wrap(foot, 190))
fig.text(0.012, 0.012, foot, color=MUTED, ha="left", va="bottom", **FOOT_KW)
foot_h = fig_text_size(foot, **FOOT_KW)[1]

# legend just above the footer
handles = [Patch(facecolor=COMP[c]["fc"], edgecolor=COMP[c].get("hc", COMP[c]["fc"]),
                 hatch=COMP[c]["hatch"], linewidth=0, label=COMP_LABEL[c]) for c in ORDER]
handles.append(Line2D([0], [0], marker="|", ls="none", ms=13, mew=2.0, color=INK,
                      label="end of generate (bold: median generate s per clip)"))
leg = fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.012, 0.012 + foot_h + 0.022),
                 ncol=3, frameon=False, fontsize=9, handlelength=1.6, handleheight=1.1,
                 columnspacing=1.6, labelcolor=INK2, borderaxespad=0)
fig.canvas.draw()
leg_top = leg.get_window_extent(renderer=renderer).y1 / fig.bbox.height

AX_B = leg_top + 0.08
TITLE_Y = 0.985 - title_h
AX_T = TITLE_Y - 0.035
AX_H = AX_T - AX_B
AXL_X, AXL_W = 0.168, 0.425
axL = fig.add_axes([AXL_X, AX_B, AXL_W, AX_H])
axR = fig.add_axes([0.700, AX_B, 0.275, AX_H])
style_axes(axL, "x")
style_axes(axR, "both")
fig.text(0.012, TITLE_Y, title_L, color=INK, ha="left", va="bottom", **TITLE_KW)
fig.text(0.648, TITLE_Y, title_R, color=INK, ha="left", va="bottom", **TITLE_KW)
fig.canvas.draw()

# ---- left: stacked horizontal bars
ys = {"orig": 5.15, "sdvg07": 3.6, "sdvg22": 2.3, "teacher": 1.1, "student": 0.0}
DIVIDER = 4.6
BAR_H = 0.46
DEC_H = 0.26  # thinner bar for the post-generate decode
XMAX = 92.0
axL.set_xlim(0, XMAX)
axL.set_ylim(-0.85, 6.1)
axL.set_xticks(np.arange(0, 91, 10))


def text_width_data(ax, s, size, weight="normal"):
    t = ax.text(0, 0, s, fontsize=size, fontweight=weight)
    w_px = t.get_window_extent(renderer=renderer).width
    t.remove()
    inv = ax.transData.inverted()
    return inv.transform((w_px, 0))[0] - inv.transform((0, 0))[0]


skipped = {}
for row in rows:
    y, x = ys[row["key"]], 0.0
    for c in ORDER:
        w = row["comps"].get(c)
        if w is None:
            continue
        stl = COMP[c]
        h = DEC_H if c == "decode" else BAR_H
        axL.barh(y, w, left=x, height=h, color=stl["fc"], edgecolor="white", linewidth=1.5, zorder=2)
        if stl["hatch"]:
            axL.barh(y, w, left=x, height=h, color="none", edgecolor=stl["hc"], hatch=stl["hatch"],
                     linewidth=0, zorder=3)
            axL.barh(y, w, left=x, height=h, color="none", edgecolor="white", linewidth=1.5, zorder=4)
        lab = f"{w:.1f}"
        if text_width_data(axL, lab, 9) + 0.5 <= w:
            kw = dict(ha="center", va="center", fontsize=9, color=stl["txt"], zorder=5)
            if stl["hatch"]:
                kw["bbox"] = dict(boxstyle="round,pad=0.08", fc=stl["fc"], ec="none")
            axL.text(x + w / 2, y, lab, **kw)
        else:
            skipped.setdefault(row["key"], []).append((c, x, w))
        if c == "commit":  # end of generate: tick + bold median generate seconds
            g_end = x + w
            axL.plot([g_end, g_end], [y - BAR_H / 2 - 0.06, y + BAR_H / 2 + 0.06], color=INK, lw=2.0,
                     solid_capstyle="butt", zorder=6)
            axL.annotate(f"{row['gen']:.1f} s" + ("*" if row["star"] else ""),
                         xy=(g_end, y + BAR_H / 2 + 0.06), xytext=(0, 1.5), textcoords="offset points",
                         ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK, zorder=6)
            row["gen_end"] = g_end
        x += w

# callouts for segments too narrow to hold their own label
callout_kw = dict(fontsize=9, color=INK2, va="center",
                  arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=1, shrinkB=0,
                                  relpos=(0, 0.5)))
for row in rows:
    y = ys[row["key"]]
    for c, x0, w in skipped.get(row["key"], []):
        xm = x0 + w / 2
        if row["key"] == "sdvg22" and c == "target":
            axL.annotate(f"14B target only {w:.1f} s (median 11 of {N_CHUNKS} chunks taken from the draft)",
                         xy=(xm, y + BAR_H / 2), xytext=(xm + 2.5, y + 0.86), ha="left", **callout_kw)
        elif row["key"] == "student" and c == "commit":
            axL.annotate(f"commit {w:.1f}", xy=(xm, y - BAR_H / 2), xytext=(xm + 4.0, y - 0.62),
                         ha="left", **callout_kw)
        else:
            raise RuntimeError(f"no callout position for {row['key']} {c}")
print("segments labelled by callout:", {k: [(c, round(w, 2)) for c, _, w in v] for k, v in skipped.items()})

# row labels (swatch + name + key numbers), right-aligned next to the axis
yax = axL.get_yaxis_transform()
axL.set_yticks([])
LINE_PT = 12.5
for row in rows:
    y = ys[row["key"]]
    n = 1 + len(row["sub"])
    top_off = LINE_PT * (n - 1) / 2
    axL.add_patch(Rectangle((-0.016, y - BAR_H / 2), 0.009, BAR_H, transform=yax, clip_on=False,
                            color=row["colour"], lw=0))
    axL.annotate(row["name"], xy=(-0.03, y), xycoords=yax, xytext=(0, top_off), textcoords="offset points",
                 ha="right", va="center", fontsize=10.5, color=INK, fontweight="bold", annotation_clip=False)
    for i, s in enumerate(row["sub"]):
        axL.annotate(s, xy=(-0.03, y), xycoords=yax, xytext=(0, top_off - LINE_PT * (i + 1) - 0.5),
                     textcoords="offset points", ha="right", va="center", fontsize=9, color=INK2,
                     annotation_clip=False)
axL.plot([-(AXL_X - 0.012) / AXL_W, 1.0], [DIVIDER, DIVIDER], transform=yax, color="#c8c8c8", lw=0.8,
         ls=(0, (4, 3)), clip_on=False)
axL.text(0.8, 5.88, "10/5 run · jar/castle × 3 seeds · original code (bf16 eager, T5 on CPU)",
         ha="left", va="center", fontsize=9, color=MUTED, style="italic")
axL.text(0.8, DIVIDER - 0.21, "bench-43 · optimized stack · median of 86 clips · under load",
         ha="left", va="center", fontsize=9, color=MUTED, style="italic")
axL.set_xlabel("seconds per clip (↓ lower is better)", color=INK)

# ---- right: quality vs generate seconds
pts = [("teacher", F["teacher"]["gen"]), ("student", F["student"]["gen"]),
       ("sdvg07", F["sdvg07"]["gen"]), ("sdvg22", F["sdvg22"]["gen"])]
q = {k: (summ[k]["quality"], summ[k]["quality_se"]) for k, _ in pts}
YTOP = 8.22
axR.set_xlim(0, 50)
axR.set_ylim(5.42, YTOP)
axR.set_yticks([5.5, 6.0, 6.5, 7.0, 7.5, 8.0])

tq, st = q["teacher"][0], F["student"]["gen"]
axR.add_patch(Rectangle((0, tq), st, YTOP - tq, facecolor="#f2f2f2", edgecolor="#9e9e9e",
                        lw=0.9, ls=(0, (3, 2)), zorder=1))
axR.axhline(tq, color="#bdbdbd", lw=0.8, ls=(0, (1, 2)), zorder=1)
axR.axvline(st, color="#bdbdbd", lw=0.8, ls=(0, (1, 2)), zorder=1)
axR.text(st + 0.9, YTOP - 0.03, "goal region: ≥ teacher quality\nand ≤ student time — empty",
         fontsize=9, color=INK2, ha="left", va="top")

cx, cy = CEIL["clip"], q["student"][0]
# measured: the two thresholds that were run
axR.plot([F["sdvg07"]["gen"], F["sdvg22"]["gen"]], [q["sdvg07"][0], q["sdvg22"][0]],
         color="#a1887f", lw=1.2, ls=(0, (3, 2)), zorder=2)
# not measured: extrapolation to the accept-all estimate
axR.plot([F["sdvg22"]["gen"], cx], [q["sdvg22"][0], cy], color="#d7ccc8", lw=1.0, ls=(0, (1, 2.5)),
         zorder=2)

for k, x in pts:
    y, se = q[k]
    axR.errorbar(x, y, yerr=se, fmt="none", ecolor=ARM[k], elinewidth=1.4, capsize=3, zorder=3)
    axR.scatter([x], [y], s=90, color=ARM[k], edgecolor="white", linewidth=2, zorder=4)
axR.scatter([cx], [cy], s=90, facecolor="white", edgecolor=ARM["other"], linewidth=1.6,
            linestyle=(0, (2, 1.5)), zorder=4)

lab = dict(fontsize=9, color=INK2)
arrow = dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=1, shrinkB=5)
axR.annotate(f"teacher 14B 4-step\n{F['teacher']['gen']:.1f} s · {q['teacher'][0]:.2f}",
             xy=(F["teacher"]["gen"], tq), xytext=(F["teacher"]["gen"] - 2.2, tq),
             ha="right", va="center", zorder=5,
             bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"), **lab)
axR.annotate(f"SDVG τ −0.7: {F['sdvg07']['gen']:.1f} s · {q['sdvg07'][0]:.2f}\n"
             f"{summ['sdvg07']['accepted']:.2f}/{N_CHUNKS} chunks from draft",
             xy=(F["sdvg07"]["gen"], q["sdvg07"][0] + q["sdvg07"][1]), xytext=(50, 7.85),
             ha="right", va="center", arrowprops=arrow, **lab)
axR.annotate(f"SDVG τ −2.2: {F['sdvg22']['gen']:.1f} s · {q['sdvg22'][0]:.2f}\n"
             f"{summ['sdvg22']['accepted']:.1f}/{N_CHUNKS} chunks from draft",
             xy=(F["sdvg22"]["gen"], q["sdvg22"][0]), xytext=(F["sdvg22"]["gen"] + 2.2, 6.63),
             ha="left", va="center", **lab)
axR.text(27.5, 7.13, "2 thresholds measured;\nlower τ → more chunks\nfrom the draft", fontsize=9,
         color=MUTED, ha="right", va="center", style="italic")
axR.annotate(f"student 1.3B 4-step\n{F['student']['gen']:.1f} s · {q['student'][0]:.2f}",
             xy=(F["student"]["gen"], q["student"][0] + q["student"][1]),
             xytext=(F["student"]["gen"] + 1.5, 6.27), ha="center", va="center", arrowprops=arrow,
             bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"), **lab)
axR.annotate(f"accept-all ceiling (estimate):\n≈ {CEIL['clip']:.0f} s, {MAX_FROM_DRAFT}/{N_CHUNKS} chunks "
             f"from draft\n(chunk 0 always from the 14B),\nstudent-level quality;\n"
             f"per chunk ≈ {CEIL['per_chunk']} s = draft {CEIL['draft']}\n"
             f"+ verify {CEIL['verify']} + commit {CEIL['commit']}",
             xy=(cx, cy), xytext=(cx + 1.8, 5.47), ha="left", va="bottom", arrowprops=arrow, **lab)

axR.set_xlabel("generate seconds per clip (↓ lower is better)", color=INK)
axR.set_ylabel("VLM visual quality, 1–10 (↑ higher is better)", color=INK)

os.makedirs(OUT, exist_ok=True)
for ext in ("png", "svg"):
    fig.savefig(os.path.join(OUT, f"f07_sdvg_analysis.{ext}"), dpi=200, facecolor="white")
print("saved", os.path.join(OUT, "f07_sdvg_analysis.png"))
for row in rows:
    print(row["key"], "gen_end", round(row["gen_end"], 2), "gen", row["gen"],
          {k: (round(v, 2) if v is not None else None) for k, v in row["comps"].items()})
print("ceiling", CEIL, "chunks", N_CHUNKS, "max from draft", MAX_FROM_DRAFT, "frames", FRAMES, "fps", VFPS,
      "max_area", MAX_AREA, "sizes", dict(size_count))
