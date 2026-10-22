#!/usr/bin/env python
"""f01_pipelines -- per-chunk pipeline schematic: teacher vs SDVG + actions vs ours (realtime 704).

Re-run:  CUDA_VISIBLE_DEVICES="" /home/hk28965/specwm/venv/bin/python paper/meeting_1012/scripts/f01_pipelines.py
Numbers are read from the raw files / facts.md (nothing estimated):
  - paper/meeting_1012/facts.md  sec. 0 (one 14B forward at 832, 4-step sigma schedule), sec. 1 (ours 2-step sigmas),
                                  sec. 2 (SDVG accept-all ceiling), sec. 5 table (704 fused + partial 1 0.761 s,
                                  under load), sec. 6 (full VAE 426 ms @704, 631 ms @832), sec. 15 table (QUIET
                                  steady s/chunk: teacher 2.939 s = 4.08 fps; 704 fused 0.630 s and ours 704
                                  0.763 s -> partial-commit cost +0.133 s/chunk in the same condition. The sec. 5 pair
                                  0.640 s (quiet) / 0.761 s (under load) differs in load, so 0.12 s is NOT the
                                  partial cost (facts sec. 1). The old "2.45 s/chunk = 4.9 fps quiet" of sec. 5 is a
                                  whole-clip average incl. the cheap early chunks, fixed 10/12)
  - paper/1012/bench43_summary.json  fps under load (teacher / SDVG tau -0.7 / ours 704), mean accepted chunks
  - lingbot-world-v2/specwm_out/bench43/*/<arm>_s*/loop.json  per-clip medians: SDVG stage times (ir_-0.7; ir_-2.2 for
    the accept-all ceiling), chunk-0 plan/score, teacher under-load steady interval (uniform4 chunk_start, same rule as
    scripts/bench43_report.py), ours cloud_s / device_s / cloud_chunk_s / lo_shape (pc1_704)
Code facts used in the labels: spec_ar_loop.py --force-first 1 (chunk 0 = target, no draft/verify);
res_split_infer.py device step = bilinear upsample + 1.3B KV commit; cloud x0 is fp32, saved as .half().
"""
import glob, json, os, re, statistics as st

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = "/home/hk28965/specwm"
OUT = ROOT + "/paper/meeting_1012"
FACTS = OUT + "/facts.md"
B43 = ROOT + "/lingbot-world-v2/specwm_out/bench43"
SUMMARY = ROOT + "/paper/1012/bench43_summary.json"
FID = "f01_pipelines"

# ----------------------------------------------------------------------------------------------- data
facts = open(FACTS, encoding="utf-8").read()


def grab(pattern, cast=float):
    m = re.search(pattern, facts)
    assert m, "facts.md: pattern not found: " + pattern
    return cast(m.group(1)) if m.lastindex == 1 else tuple(cast(g) for g in m.groups())


def table_row(prefix):
    """row of the sec. 5 table (| arm | cloud s/chunk | fps | ...)"""
    sec = facts.split("## 5.")[1].split("## 6.")[0]
    for line in sec.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == prefix:
            return float(cells[1]), float(cells[2].replace("≈", ""))
    raise KeyError(prefix)


T_FWD832 = grab(r"One 14B forward on a full 832 chunk ≈ ([\d.]+) s")
SIG4 = grab(r"σ ([\d.]+/[\d.]+/[\d.]+/[\d.]+); the model predicts", str).split("/")
SIG2 = grab(r"2 steps \(indices 0/500, σ ([\d.]+) → ([\d.]+)\)", str)
VAE704_MS = grab(r"At 704: full Wan2\.1 VAE (\d+) ms/chunk", int)
VAE832_MS = grab(r"At 832 the full VAE costs (\d+) ms/chunk", int)
SDVG_CEIL_FPS = grab(r"≈ [\d.]+ s/chunk ≈ \d+ s/clip ≈ ([\d.]+) fps")
def table_row15(prefix):
    """row of the sec. 15 quiet timing-pass table (| arm | mean steps | steady s/chunk | fps | speed-up |)"""
    sec = facts.split("## 15.")[1]
    for line in sec.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and cells[0] == prefix:
            return float(cells[2]), float(cells[3])
    raise KeyError(prefix)


# teacher quiet = STEADY s/chunk of the sec. 15 timing pass (jar + castle); NOT the sec. 5 "2.45 s / 4.9 fps",
# which is a whole-clip average (31.2 s / 13 chunks incl. the cheap early chunks; facts sec. 15 correction)
TEA_CHUNK, TEA_FPS_QUIET = table_row15("teacher 14B 4-step (uniform4)")
PC1_CHUNK, PC1_FPS = table_row("704 fused + partial 1")         # bench-43 / sec. 5, under load
# partial-commit cost in ONE condition: sec. 15 quiet timing pass (704 fused 0.630 -> ours 704 0.763 s/chunk)
FUSED_Q, FUSED_Q_FPS = table_row15("704 fused (jumps)")
PC1_Q, PC1_Q_FPS = table_row15("ours realtime mode (704 fused + partial)")

summ = json.load(open(SUMMARY))
FPS = {k: summ[k]["fps"] for k in ("teacher", "sdvg07", "ours704")}


def loops(arm):
    L = [json.load(open(f)) for f in sorted(glob.glob(f"{B43}/*/{arm}_s[0-9]*/loop.json"))]
    assert L, "no loop.json for " + arm
    return L


STAGES = ("draft", "verify", "target", "commit")
sd = loops("ir_-0.7")
N_CH = st.median(d["chunks"] for d in sd)                       # 13 chunks per 161-frame clip
sdt = {k: st.median(d["times"][k] for d in sd) for k in STAGES}
SD_GEN = st.median(d["gen_s"] for d in sd)
SD_DEC = st.median(d["decode_s"] for d in sd)
SD_ACC_MEAN = st.mean(d["accepted"] for d in sd)
SD_ACC_MED = st.median(d["accepted"] for d in sd)
SD_NONE = sum(d["accepted"] == 0 for d in sd)
# chunk 0 never goes through draft/verify (spec_ar_loop.py: --force-first 1 -> plan[0] = 'T', scores[0] = None)
SD_CH0_TARGET = sum(d["plan"][0] == "T" and d["scores"][0] is None for d in sd)
N_DRAFTED = N_CH - 1                                            # chunks that are actually drafted
sd22 = loops("ir_-2.2")                                         # accept-all ceiling uses the tau -2.2 per-chunk costs
per22 = {k: st.median(d["times"][k] for d in sd22) / N_CH for k in STAGES}
CEIL_S = per22["draft"] + per22["verify"] + per22["commit"]
tea = loops("uniform4")
TEA_GEN = st.median(d["gen_s"] for d in tea)
# same rule as scripts/bench43_report.py: median chunk interval from chunk 5 on, per clip; median over clips (under load)
TEA_LOAD = st.median(float(np.median(np.diff(d["chunk_start"])[5:])) for d in tea)
ours = loops("pc1_704")
OURS_GEN = st.median(d["gen_s"] for d in ours)                  # cloud + device step of the bench run
OURS_CLOUD = st.median(d["cloud_s"] for d in ours)              # cloud sum = the pipeline as drawn
OURS_DEV = st.median(d["device_s"] for d in ours)               # bilinear upsample + 1.3B KV commit (not needed in realtime mode)
OURS_DEV_CH = st.median(st.median(d["device_chunk_s"]) for d in ours)
# same rule as scripts/bench43_report.py: median of cloud_chunk_s from chunk 6 on, per clip; median over clips
OURS_STEADY = st.median(float(np.median(d["cloud_chunk_s"][6:])) for d in ours)
NVAL = [d["lo_shape"][0] * d["lo_shape"][1] * d["lo_shape"][2] * d["lo_shape"][3] for d in ours]
LAT_MB16 = st.median(n * 2 / 1e6 for n in NVAL)                 # fp16 (as stored with .half())
LAT_MB32 = st.median(n * 4 / 1e6 for n in NVAL)                 # fp32 (cloud x0 in res_split_infer.py)
LO_SHAPE = st.mode(tuple(d["lo_shape"]) for d in ours)

# consistency with facts.md (fails loudly if the raw files and the canonical numbers disagree)
assert abs(OURS_STEADY - PC1_CHUNK) < 0.0015, (OURS_STEADY, PC1_CHUNK)
assert abs(SD_ACC_MEAN - summ["sdvg07"]["accepted"]) < 1e-6
assert f"{SD_GEN:.1f}" == "44.6" and f"{TEA_GEN:.1f}" == "35.8", (SD_GEN, TEA_GEN)
assert FPS["ours704"] == PC1_FPS
assert SD_CH0_TARGET == len(sd), SD_CH0_TARGET
assert f"{12 / TEA_LOAD:.1f}" == f"{FPS['teacher']:.1f}", (TEA_LOAD, FPS["teacher"])
assert f"{12 / (SD_GEN / N_CH):.1f}" == f"{FPS['sdvg07']:.1f}", (SD_GEN, FPS["sdvg07"])
assert f"{OURS_GEN:.1f}" == "10.7" and f"{OURS_CLOUD:.1f}" == "9.2", (OURS_GEN, OURS_CLOUD)   # facts sec. 2
ceil_txt = grab(r"draft ([\d.]+) \+ verify ([\d.]+) \+ commit to both caches ([\d.]+)", str)
assert ceil_txt == tuple(f"{per22[k]:.2f}" for k in ("draft", "verify", "commit")), (ceil_txt, per22)
assert f"{12 / CEIL_S:.1f}" == f"{SDVG_CEIL_FPS:g}", (CEIL_S, SDVG_CEIL_FPS)

per = {k: v / N_CH for k, v in sdt.items()}                    # per chunk = per-clip median / 13 (incl. chunk 0)
per_drafted = {k: sdt[k] / N_DRAFTED for k in ("draft", "verify")}   # per drafted chunk (12 per clip)
partial_add = PC1_Q - FUSED_Q                                   # +0.133 s/chunk, both quiet (facts sec. 15)


# ----------------------------------------------------------------------------------------------- style
C_TEA, C_STU, C_SDVG, C_OURS = "#1f1f1f", "#e07b00", "#8d6e63", "#d81b60"
C_SCORER = "#8a8a8a"
C_CLOUD, C_CLOUD_TXT = "#e8f0fa", "#3f6496"
C_DEV, C_DEV_TXT = "#f6efe2", "#7d6440"
C_ACC, C_REJ = "#2f6b2f", "#a33a2a"
C_ARROW, C_GREY = "#555555", "#6b6b6b"
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path"})

W, Y0, Y1 = 133.3, -6.0, 75.0                                   # 13.33 x 8.1 in; 1 unit = 0.1 inch (y < 0: footer)
fig = plt.figure(figsize=(W / 10, (Y1 - Y0) / 10))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(Y0, Y1); ax.axis("off")
fig.patch.set_facecolor("white")
FX0, FX1 = 22.5, 110.5                                          # flow area
SX0, SX1 = 112.5, 132.3                                         # summary column
SXC = (SX0 + SX1) / 2
ROW = 1.42                                                      # text row pitch inside boxes


def region(x0, y0, x1, y1, fill, label, ltxt, ha="left"):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.0",
                                fc=fill, ec="none", zorder=0))
    lx = x0 + 0.8 if ha == "left" else x1 - 0.8
    ax.text(lx, y1 - 0.6, label, ha=ha, va="top", fontsize=9, color=ltxt, fontweight="bold", zorder=1)


def box(x0, y0, x1, y1, title, lines=(), time=None, edge=C_TEA, lw=1.7):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.8",
                                fc="white", ec=edge, lw=lw, zorder=2))
    rows = [(title, 10, "bold", "#111111")] + [(t, 9, "normal", "#333333") for t in lines]
    if time:
        rows.append((time, 10, "bold", "#111111"))
    top = (y0 + y1) / 2 + ROW * (len(rows) - 1) / 2
    for i, (t, fs, fw, c) in enumerate(rows):
        ax.text((x0 + x1) / 2, top - i * ROW, t, ha="center", va="center", fontsize=fs, fontweight=fw, color=c, zorder=3)


def arrow(p0, p1, color=C_ARROW, lw=1.4, ms=11):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=4))


def line(xs, ys, color=C_ARROW, lw=1.4):
    ax.plot(xs, ys, color=color, lw=lw, zorder=4, solid_capstyle="butt")


def bracket(x0, x1, y, text, fs=9.5, weight="bold", color="#222222", h=0.8, sub=None):
    ax.plot([x0, x0, x1, x1], [y + h, y, y, y + h], color=color, lw=1.1, zorder=3)
    ax.text((x0 + x1) / 2, y - 0.45, text, ha="center", va="top", fontsize=fs, color=color, fontweight=weight, zorder=3)
    if sub:
        ax.text((x0 + x1) / 2, y - 0.45 - 1.45, sub, ha="center", va="top", fontsize=9, color="#555555", zorder=3)


def lane_label(y_top, name, color, lines):
    n = len(lines) + 1
    ax.add_patch(Rectangle((1.2, y_top - 0.3 - ROW * n), 0.55, ROW * n, fc=color, ec="none"))
    ax.text(2.4, y_top - 1.0, name, ha="left", va="center", fontsize=11, fontweight="bold", color=color)
    for i, t in enumerate(lines):
        ax.text(2.4, y_top - 1.0 - ROW * (i + 1), t, ha="left", va="center", fontsize=9, color="#333333")


def summary(y_top, y_bot, fps, color, lines):
    ax.add_patch(FancyBboxPatch((SX0, y_bot), SX1 - SX0, y_top - y_bot, boxstyle="round,pad=0,rounding_size=0.8",
                                fc="white", ec="#d0d0d0", lw=1.0, zorder=1))
    span = 2.9 + 1.45 * (len(lines) - 1)                        # centre the block vertically
    y_big = (y_top + y_bot) / 2 + span / 2
    ax.text(SXC, y_big, f"{fps:g} fps", ha="center", va="center", fontsize=17, fontweight="bold", color=color)
    for i, (t, c, fw) in enumerate(lines):
        ax.text(SXC, y_big - 2.9 - 1.45 * i, t, ha="center", va="center", fontsize=9, color=c, fontweight=fw)


# ----------------------------------------------------------------------------------------------- header
ax.text(1.2, 72.9,
        f"Ours needs {PC1_CHUNK:.3f} s of cloud time per chunk ({PC1_FPS:g} fps); "
        f"SDVG + actions is slower than the teacher ({FPS['sdvg07']:g} vs {FPS['teacher']:g} fps)",
        ha="left", va="center", fontsize=13, fontweight="bold", color="#111111")
ax.text(1.2, 70.5,
        f"Per-chunk pipelines; 1 chunk = 3 latent frames = 12 video frames = 0.75 s, one clip = {N_CH:g} chunks. "
        "Times are per chunk unless marked “/ clip”.",
        ha="left", va="center", fontsize=10, color=C_GREY)

ly = 66.7                                                       # legend row
for i, (lab, ec) in enumerate([("14B forward", C_TEA), ("1.3B forward", C_STU), ("VAE / scorer / resize", C_SCORER)]):
    x = FX0 + i * 17.0
    ax.add_patch(FancyBboxPatch((x, ly - 0.75), 2.6, 1.5, boxstyle="round,pad=0,rounding_size=0.3", fc="white", ec=ec, lw=1.7))
    ax.text(x + 3.4, ly, lab, ha="left", va="center", fontsize=9, color="#333333")
for i, (lab, fc) in enumerate([("cloud / server GPU", C_CLOUD), ("device GPU", C_DEV)]):
    x = FX0 + 57.0 + i * 19.0
    ax.add_patch(Rectangle((x, ly - 0.75), 2.6, 1.5, fc=fc, ec="#c8c8c8", lw=0.6))
    ax.text(x + 3.4, ly, lab, ha="left", va="center", fontsize=9, color="#333333")
ax.text(SXC, 67.4, "fps ↑", ha="center", va="center", fontsize=10, fontweight="bold", color="#333333")
ax.text(SXC, 66.0, "bench-43, under load", ha="center", va="center", fontsize=9, color="#333333")

# ----------------------------------------------------------------------------------------------- lane 1: teacher
yT0, yT1 = 51.0, 64.6
region(FX0, yT0, FX1, yT1, C_CLOUD, "Cloud GPU (14B + VAE on one GPU)", C_CLOUD_TXT)
lane_label(yT1, "Teacher", C_TEA, ["14B, 4 steps, 832×480", "official schedule,", "separate clean commit"])
by0, by1 = 55.9, 62.5
bym = (by0 + by1) / 2
box(24.0, by0, 48.0, by1, "Denoise: 4 steps", ["σ " + " → ".join(SIG4), "4 × 14B forward"], f"≈{T_FWD832:.3f} s each")
arrow((48.0, bym), (51.0, bym))
box(51.0, by0, 70.5, by1, "Clean KV commit", ["t = 0 pass on the chunk", "1 × 14B forward"], f"≈{T_FWD832:.3f} s")
arrow((70.5, bym), (90.5, bym))
box(90.5, by0, 108.5, by1, "VAE decode", ["full Wan2.1, 832", "same GPU, not in fps"], f"{VAE832_MS} ms / chunk", edge=C_SCORER)
bracket(24.0, 70.5, 54.9, f"under load: {TEA_LOAD:.2f} s / chunk → {FPS['teacher']:g} fps",
        sub=f"quiet (timing pass, jar + castle): {TEA_CHUNK:.2f} s / chunk → {TEA_FPS_QUIET:.2f} fps")
summary(yT1, yT0, FPS["teacher"], C_TEA,
        [(f"{TEA_GEN:.1f} s / clip ↓", "#111111", "bold"),
         (f"quiet: {TEA_FPS_QUIET:.2f} fps", "#444444", "normal"),
         (f"({TEA_CHUNK:.2f} s / chunk, steady)", "#444444", "normal")])

# ----------------------------------------------------------------------------------------------- lane 2: SDVG
yS0, yS1 = 28.4, 49.6
region(FX0, yS0, FX1, yS1, C_CLOUD, "Cloud GPU (1.3B + 14B + VAE + ImageReward on one GPU)", C_CLOUD_TXT)
lane_label(yS1, "SDVG + actions", C_SDVG,
           ["Hu & Zhang 2604.17397,", "ported to LingBot;", "τ −0.7 (paper's τ)", "",
            "per-stage time = per-clip", f"median ÷ {N_CH:g} chunks, incl.", "chunk 0 (never drafted);",
            f"per drafted chunk (÷ {N_DRAFTED:g}):",
            f"draft {per_drafted['draft']:.2f} s, verify {per_drafted['verify']:.2f} s", "(bench-43, under load)"])
sy0, sy1 = 38.6, 47.0
sym = (sy0 + sy1) / 2
box(24.0, sy0, 35.0, sy1, "1.3B draft", ["4 steps"], f"{per['draft']:.2f} s", edge=C_STU)
arrow((35.0, sym), (37.5, sym))
box(37.5, sy0, 51.5, sy1, "Verify", ["VAE decode", "+ ImageReward", "(worst frame)"], f"{per['verify']:.2f} s", edge=C_SCORER)
dx, dh = 57.0, 3.3                                              # decision diamond
arrow((51.5, sym), (dx - dh, sym))
ax.add_patch(Polygon([(dx - dh, sym), (dx, sym + dh), (dx + dh, sym), (dx, sym - dh)], closed=True,
                     fc="white", ec=C_SDVG, lw=1.7, zorder=2))
ax.text(dx, sym, "≥ τ ?", ha="center", va="center", fontsize=10, fontweight="bold", color="#111111", zorder=3)
cx0, cx1 = 75.0, 90.0                                           # accept path -> commit
arrow((dx + dh, sym), (cx0, sym), color=C_ACC)
ax.text((dx + dh + cx0) / 2, sym + 0.7, "accept", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=C_ACC)
ax.text((dx + dh + cx0) / 2, sym - 0.7, f"{SD_ACC_MEAN:.2f} of {N_CH:g} chunks", ha="center", va="top", fontsize=9, color=C_ACC)
box(cx0, sy0, cx1, sy1, "KV commit", ["into BOTH caches", "(1.3B + 14B)"], f"{per['commit']:.2f} s")
arrow((cx1, sym), (92.5, sym))
box(92.5, sy0, 108.5, sy1, "Final VAE decode", ["whole clip, after", "the chunk loop"], f"{SD_DEC:.1f} s / clip", edge=C_SCORER)
ry = 33.0                                                       # reject path -> 14B regenerate -> commit
line([dx, dx], [sym - dh, ry], color=C_REJ)
arrow((dx, ry), (62.5, ry), color=C_REJ)
ax.text(dx - 0.8, 35.0, "reject", ha="right", va="center", fontsize=9.5, fontweight="bold", color=C_REJ)
box(62.5, 29.9, 79.5, 36.1, "14B regenerates", ["4 steps, same noise"], f"{sdt['target']:.1f} s / clip")
by_y = 31.0                                                     # chunk 0 bypass: 14B directly, no draft / verify
ax.add_patch(FancyArrowPatch((24.0, by_y), (62.5, by_y), arrowstyle="-|>", mutation_scale=11, lw=1.4, color=C_TEA,
                             linestyle=(0, (4, 2.5)), shrinkA=0, shrinkB=0, zorder=4))
ax.text(24.6, by_y + 0.55, "chunk 0: 14B directly (no draft, no verify)", ha="left", va="bottom", fontsize=9,
        fontweight="bold", color=C_TEA)
line([79.5, 82.5], [ry, ry])
arrow((82.5, ry), (82.5, sy0))
for i, t in enumerate([f"Draft kept for {SD_ACC_MEAN:.2f} of {N_CH:g}",
                       f"chunks per clip (median {SD_ACC_MED:g};",
                       f"{SD_NONE} of {len(sd)} clips keep none).",
                       f"Accept-all ceiling ≈{SDVG_CEIL_FPS:g} fps",
                       "(τ −2.2 per-chunk costs:",
                       f"{per22['draft']:.2f} + {per22['verify']:.2f} + {per22['commit']:.2f} s)"]):
    ax.text(85.5, 37.0 - 1.45 * i, t, ha="left", va="center", fontsize=9, color="#333333")
summary(yS1, yS0, FPS["sdvg07"], C_SDVG,
        [(f"{SD_GEN:.1f} s / clip ↓", "#111111", "bold"),
         (f"teacher: {TEA_GEN:.1f} s / clip", "#444444", "normal"),
         ("slower than the teacher", "#444444", "normal")])

# ----------------------------------------------------------------------------------------------- lane 3: ours
yO0, yO1 = 7.6, 26.8
cl1, dv0 = 76.0, 82.5
region(FX0, yO0, cl1, yO1, C_CLOUD, "Cloud GPU (14B only)", C_CLOUD_TXT)
region(dv0, yO0, FX1, yO1, C_DEV, "Device GPU (proposed; no 1.3B)", C_DEV_TXT, ha="right")
lane_label(yO1, "Ours (realtime)", C_OURS,
           ["cloud 14B at ≈704×384", f"(latent {LO_SHAPE[2]}×{LO_SHAPE[3]}),", "2 steps + fused commit", "+ partial commit"])
oy0, oy1 = 15.6, 24.0
oym = (oy0 + oy1) / 2
box(24.0, oy0, 35.0, oy1, "Denoise 1", [f"σ {SIG2[0]}", "1 × forward"])
arrow((35.0, oym), (37.5, oym))
box(37.5, oy0, 54.5, oy1, "Denoise 2", [f"σ {SIG2[1]}, 1 × forward;", "keep its K/V", "= fused commit"])
arrow((54.5, oym), (57.0, oym))
box(57.0, oy0, 75.0, oy1, "Partial commit", ["clean t = 0 pass on", "newest latent frame", "(≈1/3 forward)"], f"+{partial_add:.2f} s (quiet)")
bracket(24.0, 54.5, 14.3, f"{FUSED_Q:.3f} s / chunk quiet (2 forwards, fused)", fs=9, weight="normal")
bracket(24.0, 75.0, 10.9, f"cloud: {PC1_CHUNK:.3f} s / chunk → {PC1_FPS:g} fps (under load; quiet {PC1_Q:.3f} s)",
        h=1.0)
arrow((75.0, oym), (83.5, oym), color=C_OURS, lw=2.0, ms=13)  # link
ax.text(79.25, oy1 + 0.4, "latents", ha="center", va="bottom", fontsize=9, color=C_OURS, fontweight="bold")
for i, t in enumerate([f"≈{LAT_MB16:.1f} MB", "/ chunk", "(fp16)"]):
    ax.text(79.25, oy0 - 0.5 - 1.4 * i, t, ha="center", va="top", fontsize=9, color=C_OURS, fontweight="bold")
dy0, dy1 = 16.3, 23.3                                           # device boxes (shorter than the cloud ones; same centre line)
box(83.5, dy0, 98.0, dy1, "VAE decode", ["full Wan2.1", "at 704"], f"{VAE704_MS} ms / chunk", edge=C_SCORER)
arrow((98.0, oym), (100.0, oym))
box(100.0, dy0, 109.0, dy1, "Bicubic", ["upsample", "→ display"], edge=C_SCORER)
for i, t in enumerate([f"{VAE704_MS} ms (on RTX PRO 6000) < {PC1_CHUNK * 1000:.0f} ms",
                       "→ would overlap the next cloud chunk",
                       "if the device GPU is of this class.",
                       "Overlap not measured end to end;",
                       "network transfer not included."]):
    ax.text(96.5, 14.6 - 1.4 * i, t, ha="center", va="center", fontsize=9, color=C_DEV_TXT,
            fontweight="bold" if i == 0 else "normal")
summary(yO1, yO0, FPS["ours704"], C_OURS,
        [(f"{OURS_CLOUD:.1f} s / clip ↓", "#111111", "bold"),
         ("(cloud sum; see footer)", "#444444", "normal"),
         (f"cloud {PC1_CHUNK:.3f} s / chunk", "#444444", "normal"),
         ("2 + ≈1/3 forwards at 704", "#444444", "normal"),
         ("(teacher: 5 at 832)", "#444444", "normal")])

# ----------------------------------------------------------------------------------------------- footer
foot = [
    "Source: paper/meeting_1012/facts.md §0–2, §5–6, §15; per-clip medians from lingbot-world-v2/specwm_out/bench43/*/loop.json "
    f"(bench-43 = 38 StEvo tasks + 5 easy scenes × 2 seeds = {len(sd)} clips/arm).",
    "fps as in scripts/bench43_report.py (one RTX PRO 6000 Blackwell GPU, under load = 7 GPU jobs): teacher 12 / median chunk "
    f"interval from chunk 5 on ({TEA_LOAD:.2f} s; quiet {TEA_CHUNK:.2f} s, same rule, §15);",
    f"ours 12 / median cloud s per chunk from chunk 6 on ({OURS_STEADY:.3f} s); SDVG 12 / (gen s ÷ {N_CH:g}) over all chunks, "
    "incl. the cheaper early ones before the KV cache is full,",
    "which favours SDVG, so “SDVG slower than the teacher” still holds. fps excludes the final VAE decode for every arm "
    f"(the teacher's {VAE832_MS} ms / chunk would run",
    "on the same GPU); SDVG's fps includes the VAE decode inside verify. s / clip excludes prepare and final decode. "
    f"Ours {OURS_CLOUD:.1f} s / clip = cloud sum; the bench total",
    f"{OURS_GEN:.1f} s / clip adds a {OURS_DEV:.2f} s device step (bilinear upsample + a 1.3B KV commit, ≈{OURS_DEV_CH:.2f} s / chunk) "
    "that this pipeline does not need; the bench ran cloud and device steps",
    f"one after the other on one GPU. Latents {LO_SHAPE[0]}×{LO_SHAPE[1]}×{LO_SHAPE[2]}×{LO_SHAPE[3]} × 2 B ≈ {LAT_MB16:.1f} MB "
    f"(cloud x0 is fp32 in res_split_infer.py = {LAT_MB32:.2f} MB, stored as fp16). ≈1/3 forward = token share "
    "(1 of 3 latent frames).",
    f"Fused {FUSED_Q:.3f} s and partial +{partial_add:.2f} s (→ {PC1_Q:.3f} s / chunk): quiet timing pass, jar + "
    "sandcastle, no decode (facts §15); not the §5 pair 0.640 s quiet vs 0.761 s under load.",
]
for i, t in enumerate(foot):
    ax.text(1.2, 5.4 - 1.42 * i, t, ha="left", va="center", fontsize=9, color="#7a7a7a")

# ----------------------------------------------------------------------------------------------- layout check
fig.canvas.draw()
rend = fig.canvas.get_renderer()
fb = fig.bbox
bbs = [(t.get_text(), t.get_window_extent(rend)) for t in ax.texts if t.get_text().strip()]
clipped = [(s[:60], round(bb.x1 - fb.x1)) for s, bb in bbs   # every text inside the canvas (margin 12 px = 0.12 in)
           if not (bb.x0 >= fb.x0 + 6 and bb.x1 <= fb.x1 - 12 and bb.y0 >= fb.y0 + 6 and bb.y1 <= fb.y1 - 6)]
assert not clipped, ("clipped text (excess px)", clipped)
for i in range(len(bbs)):   # report text pairs whose boxes overlap by more than a sliver
    for j in range(i + 1, len(bbs)):
        a, b = bbs[i][1], bbs[j][1]
        ox = min(a.x1, b.x1) - max(a.x0, b.x0); oy = min(a.y1, b.y1) - max(a.y0, b.y0)
        if ox > 2 and oy > 4:
            print("WARN text overlap:", repr(bbs[i][0]), "<->", repr(bbs[j][0]), round(ox), round(oy))

os.makedirs(OUT + "/figs", exist_ok=True)
fig.savefig(f"{OUT}/figs/{FID}.png", dpi=200, facecolor="white")
fig.savefig(f"{OUT}/figs/{FID}.svg", facecolor="white")
print("saved", f"{OUT}/figs/{FID}.png", "and .svg")
print(dict(per_chunk=per, per_drafted=per_drafted, per22=per22, ceil_s=CEIL_S, sd_gen=SD_GEN, sd_dec=SD_DEC,
           acc_mean=SD_ACC_MEAN, acc_med=SD_ACC_MED, none=SD_NONE, ch0_target=SD_CH0_TARGET,
           tea_gen=TEA_GEN, tea_load=TEA_LOAD, ours_gen=OURS_GEN, ours_cloud=OURS_CLOUD, ours_dev=OURS_DEV,
           ours_dev_ch=OURS_DEV_CH, ours_steady=OURS_STEADY, lat_mb16=LAT_MB16, lat_mb32=LAT_MB32, lo_shape=LO_SHAPE,
           tea_chunk=TEA_CHUNK, fused_q=FUSED_Q, pc1_q=PC1_Q, pc1=PC1_CHUNK, partial_add=partial_add, fwd=T_FWD832,
           vae704=VAE704_MS, vae832=VAE832_MS, ceil=SDVG_CEIL_FPS, sig4=SIG4, sig2=SIG2, fps=FPS))
