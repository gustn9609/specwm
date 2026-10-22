#!/usr/bin/env python
"""f13_latency: per-frame action->frame latency (left) and output fps (right) per config,
measured with Chenfeng's sdv2_latency definitions (our port: scripts/wm_latency.py).

Re-run:  CUDA_VISIBLE_DEVICES="" /home/hk28965/specwm/venv/bin/python \
         /home/hk28965/specwm/paper/meeting_1012/scripts/f13_latency.py

Data sources (paths relative to /home/hk28965/specwm):
  - paper/meeting_1012/facts.md sections 13 and 15 (canonical tables; every number drawn here is
    asserted to match them to rounding)
  - lingbot-world-v2/specwm_out/sdvg_profile/latency/<label>/{summary.json, frames.csv}
    labels orig_draft, orig_target, orig_sdvg07, orig_sdvg22 (original stack, run fast0) and
    opt_draft, opt_target, opt_sdvg07, opt_sdvg22 (optimized stack, run fast1)
  - lingbot-world-v2/specwm_out/timing_pass/latency/<label>/{summary.json, frames.csv}
    OURS rows: ours704 (realtime mode), ours832 (quality mode) = operating modes, drawn prominently;
    c704_fused, c704_sep, u2_832_sep, c576_pc1 = other variants (not operating modes), drawn light.
    ours704 / ours832 / c704_sep have n = 5 clips (castle s1234 excluded: cold compile), the others 6.
  - lingbot-world-v2/specwm_out/timing_pass/<task>/<arm>_s<seed>/loop.json (lo_shape = decoded size;
    gen_s of the excluded cold clips, quoted in the footer)
  - paper/meeting_1012/data/f02_boundary_counts.json (boundary-jump share of 704 fused vs fused + partial)
  - paper/meeting_1012/data/e_steady_fps.json (opt_target: the teacher's steady output fps with the streamed decode,
    3.80, and whole-clip / steady = 1.095; used only to keep the teacher's fps on one basis in the table note)
  - paper/meeting_1012/facts.md section 2 (bench-43 SDVG tau -2.2: 6.4 / 6.7 steady vs teacher 3.9 fps, 10.1 of 13
    chunks from the draft): the scope of the "SDVG is slower" claim, which holds on jar + sandcastle only
Outputs: figs/f13_latency.{png,svg}, data/f13_latency_table.md. No GPU is used.

Left panel: one row per config, Gaussian KDE of server_ms over all frames of the config's clips (same
kernel and bandwidth rule as sdv2_latency.gaussian_kde; each row scaled to its own peak), black
tick = p50, dashed tick + label = p99, red line = 750 ms deadline. Right panel: output fps per
config (mean over the clips, summary.json fps_mean; per clip fps = (frames - 1) / (last - first
ready), as scripts/wm_latency.py) as horizontal bars on the same rows, dots = per-clip values,
red line = 16 fps playback.

Checks before drawing: facts.md §13 tables (profile + ours modes) and §15 streaming line == summary.json;
frames.csv == summary.json; per-clip pairing SDVG vs teacher; SDVG low mode == chunk 0 + accepted draft
chunks (loop.json plan); no cold-start clip (gen_s > 1.5x median of the other clips) in any row (a row
with one would be flagged with a dagger); title claims (ours realtime p50 below every baseline and mode; SDVG
slower than the teacher, scoped to jar + sandcastle with the bench-43 contrast from facts section 2).
Basis rule: output fps (whole-clip mean) is compared only with output fps; steady generation fps (facts section 15)
only with steady fps.
"""
import csv
import json
import math
import os
import re
import textwrap

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

ROOT = "/home/hk28965/specwm"
PKG = os.path.join(ROOT, "paper/meeting_1012")
FACTS = os.path.join(PKG, "facts.md")
PROFILE = os.path.join(ROOT, "lingbot-world-v2/specwm_out/sdvg_profile/latency")
TP_ROOT = os.path.join(ROOT, "lingbot-world-v2/specwm_out/timing_pass")
TIMING = os.path.join(TP_ROOT, "latency")
BOUNDARY = os.path.join(PKG, "data/f02_boundary_counts.json")
OUT_FIG = os.environ.get("F13_OUT_FIG", os.path.join(PKG, "figs"))
OUT_TAB = os.environ.get("F13_OUT_TAB", os.path.join(PKG, "data/f13_latency_table.md"))

BUDGET_MS = 750.0     # latency deadline per frame: one chunk of playback (12 frames at 16 fps)
PLAYBACK_FPS = 16.0

# ---------------------------------------------------------------- style (meeting colours)
C = {"teacher": "#1f1f1f", "student": "#e07b00", "sdvg07": "#8d6e63", "sdvg22": "#c49a6c",
     "ours832": "#7b4bd1", "ours704": "#d81b60", "parity": "#1565c0"}
C_RT = "#c62828"
INK, INK2, MUTED, GRID = "#1f1f1f", "#4a4a4a", "#7a7a7a", "#e6e6e6"
BAND = "#f3f3f3"  # background band behind our operating modes
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
    "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "axes.edgecolor": "#9a9a9a", "axes.linewidth": 0.8, "hatch.linewidth": 0.8,
    "svg.fonttype": "none",
})
NB = "\u00a0"  # non-breaking space (figure text only; the table replaces it with a plain space)
f1 = lambda x: f"{x:.1f}"  # noqa: E731


def lighten(hex_c, f=0.55):
    r, g, b = to_rgb(hex_c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def fmt_miss(x):  # same rounding in the figure, the table and facts.md: "100 %" not "100.0 %"
    return "100 %" if round(x, 1) == 100 else f"{x:.1f} %"


# ---------------------------------------------------------------- configs
# variant: "orig" = original stack (lighter fill, dashed outline); "opt" = optimized stack (solid);
#          "alt" = an ours variant that is not an operating mode (lighter fill, dotted outline, compact row)
PROFILE_ROWS = [  # label, row name, colour key, variant, table name
    ("orig_draft", "draft only (1.3B)", "student", "orig", "original: draft only (1.3B)"),
    ("orig_target", "target only (14B 4-step)", "teacher", "orig", "original: target only (14B 4-step)"),
    ("orig_sdvg07", "SDVG τ −0.7", "sdvg07", "orig", "original: SDVG τ −0.7"),
    ("orig_sdvg22", "SDVG τ −2.2", "sdvg22", "orig", "original: SDVG τ −2.2"),
    ("opt_draft", "student (draft only, 1.3B)", "student", "opt", "optimized: draft only (= student)"),
    ("opt_target", "teacher (target only, 14B)", "teacher", "opt", "optimized: target only (= teacher)"),
    ("opt_sdvg07", "SDVG τ −0.7", "sdvg07", "opt", "optimized: SDVG τ −0.7"),
    ("opt_sdvg22", "SDVG τ −2.2", "sdvg22", "opt", "optimized: SDVG τ −2.2"),
]
MODE_ROWS = [
    ("ours704", "ours: realtime mode (704)", "ours704", "opt",
     "ours: realtime mode (704, 2-step, fused + partial commit)"),
    ("ours832", "ours: quality mode (832)", "ours832", "opt",
     "ours: quality mode (832, 2-step, fused + partial commit)"),
]
ALT_ROWS = [
    ("c704_fused", "704 fused, no partial commit", "ours704", "alt", "704 2-step, fused, no partial commit (c704_fused)"),
    ("c704_sep", "704, separate commit", "ours704", "alt", "704 2-step, separate commit (c704_sep)"),
    ("u2_832_sep", "832, separate commit", "ours832", "alt", "832 2-step, separate commit (u2_832_sep)"),
    ("c576_pc1", "576 fused + partial", "ours704", "alt", "576 2-step, fused + partial commit (c576_pc1)"),
]
# teacher-parity mode (cw07_safe, spec_early) has no row: spec_early records no per-frame ready times
# (no --stream-decode support). Stated in the footer and the table, not only here.
TASK_SHORT = {"sand_filling_up_jar": "jar", "sandcastle_wave_impact": "castle"}


def load(base, label):
    sp, fp = os.path.join(base, label, "summary.json"), os.path.join(base, label, "frames.csv")
    assert os.path.exists(sp) and os.path.exists(fp), f"missing {sp} / {fp}"
    s = json.load(open(sp, encoding="utf-8"))
    with open(fp, newline="") as h:
        rows = [r for r in csv.DictReader(h) if r.get("server_ms")]
    v = np.array([float(r["server_ms"]) for r in rows])
    seg = np.array([int(r["segment"]) for r in rows])
    assert len(v) and s.get("frames") == len(v), (label, len(v), s.get("frames"))
    # frames.csv and summary.json must describe the same data
    for q in (50, 90, 99):
        assert abs(np.percentile(v, q) - s["server_ms"][f"p{q}"]) < 0.01, (label, q)
    assert abs(np.mean(v > BUDGET_MS) - s["server_ms"]["miss_rate"]) < 1e-9, label
    assert abs(np.mean([c["fps"] for c in s["per_clip"]]) - s["fps_mean"]) < 1e-6, label
    # frames.csv is written clip by clip in per_clip order: slice it into clips (each starts at chunk 0)
    i = 0
    for c in s["per_clip"]:
        assert seg[i] == 0 and c["frames"] > 0, (label, c["dir"])
        c["p50_ms"] = float(np.percentile(v[i:i + c["frames"]], 50))
        i += c["frames"]
    assert i == len(v), label
    return dict(sum=s, ms=v)


def clip_key(c):  # (task, seed) of a per_clip entry, e.g. ("sand_filling_up_jar", "4321")
    b = c["dir"].rstrip("/")
    task = [t for t in TASK_SHORT if t in b]
    assert len(task) == 1, b
    return task[0], b.rsplit("_s", 1)[1]


def cold_clips(per_clip):
    """Clips whose generation took > 1.5x the median of the other clips (cold compile / warm-up miss)."""
    out = []
    for i, c in enumerate(per_clip):
        others = [d["gen_s"] for j, d in enumerate(per_clip) if j != i and d.get("gen_s")]
        if c.get("gen_s") and others and c["gen_s"] > 1.5 * float(np.median(others)):
            out.append((c, float(np.min(others)), float(np.max(others))))
    return out


def make_row(base, spec, kind):
    label, name, ckey, variant, tname = spec
    d = load(base, label)
    s = d["sum"]
    cold = cold_clips(s["per_clip"])
    if cold:
        name, tname = name + " †", tname + " †"
    return dict(label=label, name=name, tname=tname, colour=C[ckey], variant=variant, kind=kind, ms=d["ms"],
                cold=cold, fps=s["fps_mean"], p50=s["server_ms"]["p50"], p90=s["server_ms"]["p90"],
                p99=s["server_ms"]["p99"], miss=100 * s["server_ms"]["miss_rate"],
                first=s["first_frame_ms_mean"], gen=s["gen_s_mean"], clips=s["clips"],
                frames=s["frames"], prep=s.get("prepare_s_mean"), per_clip=s["per_clip"])


prof = [make_row(PROFILE, sp, "base") for sp in PROFILE_ROWS]
modes = [make_row(TIMING, sp, "mode") for sp in MODE_ROWS]
alts = [make_row(TIMING, sp, "alt") for sp in ALT_ROWS]
ours = modes + alts
P = {r["label"]: r for r in prof + ours}

# ---------------------------------------------------------------- facts.md cross-checks
facts = open(FACTS, encoding="utf-8").read()
sec13 = facts.split("## 13.")[1].split("## 14.")[0]
sec15 = facts.split("## 15.")[1]


def fnum(s):
    return float(re.sub(r"[^0-9.]", "", s))


TOL = dict(fps=0.051, p50=0.51, p90=0.51, p99=0.51, miss=0.051, first=0.51, gen=0.051)


def check_row(r, line):
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    want = dict(fps=fnum(cells[1]), p50=fnum(cells[2]), p90=fnum(cells[3]), p99=fnum(cells[4]),
                miss=fnum(cells[5]), first=fnum(cells[6]), gen=fnum(cells[7]))
    for k, v in want.items():
        assert abs(r[k] - v) <= TOL[k], (r["label"], k, r[k], v, line)


trows = [l for l in sec13.splitlines() if l.startswith("| original:") or l.startswith("| optimized:")]
assert len(trows) == 8, len(trows)
for spec, line in zip(PROFILE_ROWS, trows):
    check_row(P[spec[0]], line)
for lab, key in (("ours704", "| ours realtime mode (704"), ("ours832", "| ours quality mode (832")):
    line = [l for l in sec13.splitlines() if l.startswith(key)]
    assert len(line) == 1, key
    check_row(P[lab], line[0])
# §15 streaming line: "c704_sep 9.3 fps p50 1484 ms; ..."
for lab in ("c704_sep", "c704_fused", "u2_832_sep", "c576_pc1"):
    m = re.search(rf"{lab} ([0-9.]+) fps p50 ([0-9]+) ms", sec15)
    assert m, lab
    assert abs(P[lab]["fps"] - float(m.group(1))) <= 0.051 and abs(P[lab]["p50"] - float(m.group(2))) <= 0.51, lab
m = re.search(r"c576_pc1 [^;]*?\(miss > 750 ms ([0-9]+) %\)", sec15)
assert m and abs(P["c576_pc1"]["miss"] - float(m.group(1))) < 0.5
print("facts.md §13 / §15 match the 8 profile + 6 ours summaries")


def s15_fps(prefix):  # quiet steady generation-only fps (decode excluded) from the facts §15 table
    line = [l for l in sec15.splitlines() if l.startswith(prefix)]
    assert len(line) == 1, prefix
    return float(line[0].strip("|").split("|")[3].strip())


GEN_FPS = {"teacher": s15_fps("| teacher 14B 4-step (uniform4)"),
           "ours832": s15_fps("| ours quality mode (832 fused + partial)"),
           "ours704": s15_fps("| ours realtime mode (704 fused + partial)"),
           "parity": s15_fps("| ours teacher-parity mode")}
assert (GEN_FPS["teacher"], GEN_FPS["ours832"], GEN_FPS["ours704"], GEN_FPS["parity"]) == (4.08, 9.74, 15.73, 6.80)

# ---------------------------------------------------------------- protocol facts from the raw files
assert {r["clips"] for r in prof} == {6} and {r["frames"] for r in prof} == {918}
FR_PER_CLIP = 153
for r in prof + ours:
    assert all(c["frames"] == FR_PER_CLIP for c in r["per_clip"]), r["label"]
    assert r["frames"] == FR_PER_CLIP * r["clips"], r["label"]
TASKS = ["sand_filling_up_jar", "sandcastle_wave_impact"]
SEEDS = ["1234", "4321", "9999"]
FULL = {(t, s) for t in TASKS for s in SEEDS}
for r in prof:
    assert {clip_key(c) for c in r["per_clip"]} == FULL, r["label"]
N5 = [r for r in ours if r["clips"] == 5]
assert sorted(r["label"] for r in N5) == ["c704_sep", "ours704", "ours832"], [r["label"] for r in N5]
assert all(r["clips"] == 6 for r in ours if r not in N5)
EXCL = {}  # label -> (gen_s of the excluded clip, min, max gen_s of the kept clips)
for r in N5:
    miss = FULL - {clip_key(c) for c in r["per_clip"]}
    assert miss == {("sandcastle_wave_impact", "1234")}, (r["label"], miss)
    L = json.load(open(os.path.join(TP_ROOT, "sandcastle_wave_impact", f"{r['label']}_s1234", "loop.json")))
    kept = [c["gen_s"] for c in r["per_clip"]]
    assert L["gen_s"] > 1.5 * max(kept), r["label"]  # it was a cold start, not a slow outlier
    EXCL[r["label"]] = (L["gen_s"], min(kept), max(kept))
for r in ours:
    if r not in N5:
        assert {clip_key(c) for c in r["per_clip"]} == FULL, r["label"]
assert not any(r["cold"] for r in prof + ours), [(r["label"], r["cold"]) for r in prof + ours if r["cold"]]
prep_lo = min(r["prep"] for r in prof)  # per-config means, as in facts.md §13 (23–27 s)
prep_hi = max(r["prep"] for r in prof)

# decoded frame size per ours arm = loop.json lo_shape (latent H, W) x 8, no device stage (--no-device)
SIZE = {}
for r in ours:
    hw = set()
    for c in r["per_clip"]:
        task, seed = clip_key(c)
        L = json.load(open(os.path.join(TP_ROOT, task, f"{r['label']}_s{seed}", "loop.json")))
        assert L["no_device"] and L["stream_decode"] == 1, r["label"]
        hw.add((8 * L["lo_shape"][2], 8 * L["lo_shape"][3]))
    SIZE[r["label"]] = hw


def size_str(lab):
    hs, ws = sorted({h for h, _ in SIZE[lab]}), sorted({w for _, w in SIZE[lab]})
    rng = lambda v: f"{v[0]}" if len(v) == 1 else f"{v[0]}–{v[-1]}"  # noqa: E731
    return f"{rng(hs)}×{rng(ws)}"


# boundary jumps (facts §5) for the 704 fused variant vs the realtime mode
BJ = json.load(open(BOUNDARY, encoding="utf-8"))["arms"]
bj_fused = 100 * BJ["704 fused"]["boundary_k"] / BJ["704 fused"]["boundary_n"]
bj_pc1 = 100 * BJ["704 fused + partial 1"]["boundary_k"] / BJ["704 fused + partial 1"]["boundary_n"]
assert (round(bj_fused), round(bj_pc1)) == (12, 4)

# ---------------------------------------------------------------- SDVG bimodality (checked in loop.json)
# low mode = chunk 0 (always target, no draft/verify) + chunks taken from the draft; high mode = rejected
# chunks (draft + verify + 14B regenerate). Verified per clip from loop.json plan / chunk_start / frame_ready.
LOW = {}
for lab in ("orig_sdvg07", "orig_sdvg22", "opt_sdvg07", "opt_sdvg22"):
    lo, hi = [], []
    for c in P[lab]["per_clip"]:
        L = json.load(open(os.path.join(ROOT, c["dir"], "loop.json"), encoding="utf-8"))
        assert L["plan"][0] == "T" and L["plan"].count("D") == c["accepted"], (lab, c["dir"])
        for _, ch, t in L["frame_ready"]:
            (lo if ch == 0 or L["plan"][ch] == "D" else hi).append(1000 * (t - L["chunk_start"][ch]))
    assert max(lo) < min(hi), lab  # the two modes do not overlap
    assert len(lo) == 6 * 9 + 12 * sum(c["accepted"] for c in P[lab]["per_clip"]), lab
    LOW[lab] = len(lo)

# ---------------------------------------------------------------- claims (checked, then written)
o_t, o_s, stud = P["opt_target"], [P["opt_sdvg07"], P["opt_sdvg22"]], P["opt_draft"]
g_t, g_s = P["orig_target"], [P["orig_sdvg07"], P["orig_sdvg22"]]
rt, qm = P["ours704"], P["ours832"]
assert all(r["fps"] < o_t["fps"] for r in o_s) and all(r["fps"] < g_t["fps"] for r in g_s)
assert all(r["p50"] > o_t["p50"] for r in o_s)
# "lowest action->frame latency of all measured baselines and modes": every profile row and the quality mode
assert all(rt["p50"] < r["p50"] for r in prof + [qm])
assert rt["fps"] >= max(r["fps"] for r in prof + [qm])
# only two non-mode variants are lower, and they are named in the subtitle
lower_alts = sorted([r for r in alts if r["p50"] < rt["p50"]], key=lambda r: r["p50"])
assert [r["label"] for r in lower_alts] == ["c576_pc1", "c704_fused"], [r["label"] for r in lower_alts]
min_miss = min(r["miss"] for r in prof + ours)
assert min_miss == P["c576_pc1"]["miss"] and min_miss > 50

# per-clip pairing (same task and seed): where does SDVG beat the teacher of the same stack?
flips, stack_of = {}, {}
for stack, T, rows in (("original", g_t, g_s), ("optimized", o_t, o_s)):
    tk = {clip_key(c): c for c in T["per_clip"]}
    for S, tau in zip(rows, ("−0.7", "−2.2")):
        f_fps, f_p50 = [], []
        for c in S["per_clip"]:
            stack_of[id(c)] = stack
            t = tk[clip_key(c)]
            if c["fps"] > t["fps"]:
                f_fps.append((c, t))
            elif c["p50_ms"] < t["p50_ms"]:
                f_p50.append((c, t))
        flips[(stack, tau)] = dict(fps=f_fps, p50_only=f_p50)
for stack in ("original", "optimized"):  # τ −0.7: slower on every clip, by fps and by p50
    assert not flips[(stack, "−0.7")]["fps"] and not flips[(stack, "−0.7")]["p50_only"]


def cname(c):
    task, seed = clip_key(c)
    return f"{stack_of[id(c)]} {TASK_SHORT[task]} s{seed}"


fps_flips = flips[("optimized", "−2.2")]["fps"] + flips[("original", "−2.2")]["fps"]
p50_flips = flips[("optimized", "−2.2")]["p50_only"] + flips[("original", "−2.2")]["p50_only"]
assert fps_flips
min_acc = min(c["accepted"] for c, _ in fps_flips + p50_flips)
n_fl = {st: len(flips[(st, "−2.2")]["fps"]) for st in ("original", "optimized")}
per_clip_txt = (f"τ{NB}−0.7 is slower on 6/6 clips in both stacks. τ{NB}−2.2 is faster on "
                f"{n_fl['optimized']}/6 (optimized) and {n_fl['original']}/6 (original) clips, only with ≥{NB}{min_acc} of "
                "13 chunks taken from the draft: "
                + "; ".join(f"{cname(c)}, {c['accepted']} of 13: {c['fps']:.2f} vs {t['fps']:.2f}{NB}fps, p50 "
                            f"{c['p50_ms']:.0f} vs {t['p50_ms']:.0f}{NB}ms" for c, t in fps_flips)
                + "".join(f"; {cname(c)} ({c['accepted']} of 13) has a lower p50 ({c['p50_ms']:.0f} vs "
                          f"{t['p50_ms']:.0f}{NB}ms) but lower fps ({c['fps']:.2f} vs {t['fps']:.2f})"
                          for c, t in p50_flips) + ".")

# SDVG scope: slower than the teacher on THESE two hard prompts only. On bench-43 τ −2.2 takes far more chunks
# from the draft and is faster than the teacher (facts §2), so the claim is always written with its scope.
acc = {st: [c["accepted"] for c in P[f"{pre}_sdvg22"]["per_clip"]] for st, pre in (("original", "orig"),
                                                                                    ("optimized", "opt"))}
assert acc == {"original": [2, 4, 1, 7, 5, 4], "optimized": [2, 10, 2, 5, 7, 5]}, acc  # facts §13
acc_mean = {st: float(np.mean(v)) for st, v in acc.items()}
assert f"mean {acc_mean['original']:.1f}" in sec13 and f"mean {acc_mean['optimized']:.1f}" in sec13
sec2 = facts.split("## 2.")[1].split("## 3.")[0]
b43_t = [l for l in sec2.splitlines() if l.startswith("| teacher 14B 4-step |")]
b43_s = [l for l in sec2.splitlines() if l.startswith("| SDVG + actions τ −2.2 |")]
assert len(b43_t) == 1 and len(b43_s) == 1
B43 = dict(teacher=float(b43_t[0].split("|")[2]), sdvg22=float(b43_s[0].split("|")[2]),
           acc22=float(b43_s[0].strip().strip("|").split("|")[-1].split("(")[0]))
m = re.search(r"τ −2\.2 [0-9.]+ s/chunk → ([0-9.]+) fps", sec2)
assert m
B43["sdvg22_steady"] = float(m.group(1))
assert (B43["teacher"], B43["sdvg22"], B43["sdvg22_steady"], B43["acc22"]) == (3.9, 6.4, 6.7, 10.1), B43
B43["dq22"] = b43_s[0].split("|")[5].strip()
assert B43["dq22"] == "−0.84 ± 0.10", B43

title = (f"Ours realtime mode has the lowest action→frame latency of all measured baselines and modes:\n"
         f"p50 {rt['p50'] / 1000:.2f}{NB}s at {f1(rt['fps'])}{NB}fps on one GPU (teacher "
         f"{o_t['p50'] / 1000:.2f}{NB}s / {f1(o_t['fps'])}{NB}fps, student {stud['p50'] / 1000:.2f}{NB}s / "
         f"{f1(stud['fps'])}{NB}fps);\non jar{NB}+{NB}sandcastle SDVG is slower than the teacher at both τ "
         f"({f1(min(r['fps'] for r in o_s))}–{f1(max(r['fps'] for r in o_s))} vs {f1(o_t['fps'])}{NB}fps)")
fc, ff = P["c704_fused"], P["c576_pc1"]
sub = [
    (f"Ours, one GPU, full-VAE decode streamed on the same GPU: realtime mode {f1(rt['fps'])}{NB}fps output, p50 "
     f"{rt['p50']:.0f}{NB}ms, first frame {rt['first']:.0f}{NB}ms; quality mode {f1(qm['fps'])}{NB}fps, p50 "
     f"{qm['p50']:.0f}{NB}ms (output fps = whole-clip mean, not steady). Their generation alone (quiet steady, "
     f"decode excluded; facts{NB}§15) runs at {GEN_FPS['ours704']:.1f} / {GEN_FPS['ours832']:.1f}{NB}fps; the gap is "
     f"the full-VAE decode sharing the GPU."),
    (f"Lighter rows are not operating modes: 704 fused without partial commit has lower latency (p50 "
     f"{fc['p50']:.0f}{NB}ms, {f1(fc['fps'])}{NB}fps) but {bj_fused:.0f}{NB}% boundary jumps vs {bj_pc1:.0f}{NB}% "
     f"(facts{NB}§5); 576 fused + partial is the fastest measured (p50 {ff['p50']:.0f}{NB}ms, {f1(ff['fps'])}{NB}fps) at "
     f"a lower resolution ({size_str('c576_pc1')})."),
    (f"On jar{NB}+{NB}sandcastle, SDVG on the same stack is slower than the teacher on average "
     f"({f1(min(r['fps'] for r in o_s))}–{f1(max(r['fps'] for r in o_s))} vs {f1(o_t['fps'])}{NB}fps optimized, "
     f"{f1(min(r['fps'] for r in g_s))}–{f1(max(r['fps'] for r in g_s))} vs {f1(g_t['fps'])} original): these hard "
     f"prompts let τ{NB}−2.2 take only {acc_mean['optimized']:.1f} (optimized) / {acc_mean['original']:.1f} (original) "
     f"of 13 chunks from the draft vs {B43['acc22']:.1f} on bench-43, where it is faster than the teacher "
     f"({B43['sdvg22']:.1f} whole-clip / {B43['sdvg22_steady']:.1f} steady vs {B43['teacher']:.1f}{NB}fps, under load; "
     f"facts{NB}§2). Per clip here τ{NB}−0.7 is slower on 6/6 clips, τ{NB}−2.2 faster on {n_fl['optimized']}/6 (optimized) "
     f"and {n_fl['original']}/6 (original), only with ≥{NB}{min_acc} of 13 chunks from the draft. Every config misses "
     f"the 750{NB}ms deadline on ≥{NB}{math.floor(min_miss)}{NB}% of frames (student and ours realtime "
     f"{f1(stud['miss'])}{NB}%)."),
]
assert f1(stud["miss"]) == f1(rt["miss"])
# teacher on one basis (steady, optimized profile run, decode streamed on the same GPU; facts §13)
EST = json.load(open(os.path.join(PKG, "data/e_steady_fps.json"), encoding="utf-8"))["opt_target"]
assert abs(EST["fps_mean_whole_clip"] - o_t["fps"]) < 1e-3 and f"target {EST['steady_fps']:.2f}" in sec13

# ---------------------------------------------------------------- rows layout (data y units)
groups = [  # header, rows, band
    ("original stack · bf16 eager (the code of the 10/5 slide) · n = 6 clips", prof[:4], False),
    ("optimized stack · FP8 W8A8 + SageAttention 2 + torch.compile + ring KV · n = 6 clips", prof[4:], False),
    ("ours · operating modes · optimized stack, timing pass (separate run) · n = 5 clips", modes, True),
    ("ours · other variants, not operating modes · same timing pass", alts, False),
]
HDR_H = 0.62
GEOM = {"base": (1.0, 0.80), "mode": (1.0, 0.80), "alt": (0.80, 0.60)}  # row height, KDE height
y, layout, dividers, bands = 0.0, [], [], []
for gi, (hdr, rows, band) in enumerate(groups):
    if gi:
        y -= 0.22  # keep the divider clear of the previous row's second label line
        dividers.append(y)
        y -= 0.04
    g_top = y
    layout.append(("hdr", hdr, y - 0.30, band))
    y -= HDR_H
    for r in rows:
        rh, kh = GEOM[r["kind"]]
        r["kde_h"] = kh
        r["base"] = y - (rh - 0.08)
        r["mid"] = r["base"] + 0.45 * kh
        layout.append(("row", r, r["base"], band))
        y -= rh
    if band:
        bands.append((g_top + 0.02, y - 0.18))
Y_MIN, Y_MAX = y + 0.02, 0.0

# ---------------------------------------------------------------- figure
# The figure height is derived from the measured text blocks, so every config row keeps ROW_IN inches.
FIG_W = 13.33
ROW_IN = 0.34  # inches per data-y unit (one full config row = 1.0)
XL = 0.012
SUB_KW = dict(fontsize=10, color=INK2, linespacing=1.35)
TITLE_KW = dict(fontsize=13.5, fontweight="bold", color=INK, linespacing=1.25)
title_w = title  # explicit line break after the claim
sub_w = "\n".join(l for s in sub for l in textwrap.wrap(s, 178, break_on_hyphens=False))

low_note = (f"SDVG rows are bimodal: the low mode is chunk 0 (always target) plus the chunks taken from the draft "
            f"(original {LOW['orig_sdvg07']} / {LOW['orig_sdvg22']}, optimized {LOW['opt_sdvg07']} / "
            f"{LOW['opt_sdvg22']} of {FR_PER_CLIP * 6} frames at τ{NB}−0.7 / −2.2); the high mode is the rejected chunks "
            "(draft + verify + 14B regenerate). ")
ex = [EXCL[l] for l in ("ours704", "ours832", "c704_sep")]
foot = (
    f"Protocol (all rows): jar + sandcastle × seeds {'/'.join(SEEDS)}, 161 frames ({FR_PER_CLIP} output frames per clip), "
    "empty text prompt, one RTX PRO 6000 Blackwell GPU per run, prepare excluded. Streaming decode: every committed "
    "chunk is decoded right away with the full Wan2.1 VAE on a side CUDA stream of the SAME GPU, so decode competes "
    "with generation. server_ms = frame ready in host memory − start of its chunk (action consumed); output fps = "
    "(frames − 1) / (last − first ready) per clip (whole clip, not steady), mean over the clips. KDE: Gaussian, "
    "bandwidth σ·n^(−1/5) as in sdv2_latency, each row scaled to its own peak. "
    "Profile rows (scripts/sdvg_profile.sh, n = 6 each): original and optimized stacks ran concurrently on two separate "
    "GPUs, no other generation jobs, a vLLM Qwen3-VL-32B judge on another GPU; prepare "
    f"{prep_lo:.0f}–{prep_hi:.0f}{NB}s there (both models, T5 on CPU). " + low_note +
    "SDVG = Hu & Zhang (arXiv 2604.17397) ported to action-conditioned LingBot (ImageReward verifier). "
    "Ours rows (scripts/timing_pass.sh, a later separate run, 2 lanes on 2 GPUs, no other generation jobs, VLM "
    "judges on other GPUs): "
    "specwm.res_split_infer --stream-decode 1 --no-device = cloud 14B + streaming decode on the same GPU, no device "
    f"stage, frames decoded at the cloud resolution (realtime {size_str('ours704')}, quality {size_str('ours832')}, "
    f"576 variant {size_str('c576_pc1')}). n = 5 clips for ours realtime, ours quality and 704 separate commit: castle "
    "s1234 excluded because the warm-up covered only jar's latent shape and that clip paid a cold torch.compile "
    f"(gen {ex[0][0]:.1f} / {ex[1][0]:.1f} / {ex[2][0]:.1f}{NB}s vs {ex[0][1]:.1f}–{ex[0][2]:.1f} / "
    f"{ex[1][1]:.1f}–{ex[1][2]:.1f} / {ex[2][1]:.1f}–{ex[2][2]:.1f}{NB}s for the kept clips); n = 6 for the other "
    "variants (per-task warm-up). Teacher-parity mode (cw07_safe) has no row: spec_early has no streaming decode, so "
    f"it records no per-frame ready times (quiet steady generation {GEN_FPS['parity']:.1f}{NB}fps, facts §15); "
    "Matrix-Game 2.0 / Waypoint log no per-frame ready times either. Source: "
    "lingbot-world-v2/specwm_out/{sdvg_profile,timing_pass}/latency/<label>/{frames.csv, summary.json}; "
    "paper/meeting_1012/facts.md §2, §5, §13, §15; data/e_steady_fps.json; table: paper/meeting_1012/data/f13_latency_table.md."
)
FOOT_KW = dict(fontsize=8.5, color=MUTED, linespacing=1.38)
foot_w = "\n".join(textwrap.wrap(foot, 200, break_on_hyphens=False))

# legend handles
key_handles = [Patch(fc=C["student"], ec=C["student"], label="1.3B draft / student"),
               Patch(fc=C["teacher"], ec=C["teacher"], label="14B target / teacher"),
               Patch(fc=C["sdvg07"], ec=C["sdvg07"], label="SDVG τ −0.7"),
               Patch(fc=C["sdvg22"], ec=C["sdvg22"], label="SDVG τ −2.2"),
               Patch(fc=C["ours704"], ec=C["ours704"], label="ours: realtime mode (704)"),
               Patch(fc=C["ours832"], ec=C["ours832"], label="ours: quality mode (832)")]
style_handles = [
    Patch(fc=lighten("#757575"), ec="#757575", ls=(0, (3, 2)), lw=1.2, label="original stack (lighter, dashed)"),
    Patch(fc="#757575", ec="#757575", lw=1.2, label="optimized stack (solid)"),
    Patch(fc=lighten("#757575"), ec="#757575", ls=(0, (1, 1.5)), lw=1.2, label="ours, other variants (lighter, dotted)"),
]
mark_handles = [
    Line2D([0], [0], color=INK, ls="none", marker="|", ms=12, mew=1.8, label="p50 (black tick)"),
    Line2D([0], [0], color=INK2, lw=1.4, ls=(0, (3, 2)), label="p99 (dashed tick, labelled)"),
    Line2D([0], [0], color=C_RT, lw=1.4, ls=(0, (6, 3)),
           label="750 ms deadline per frame (a) / 16 fps playback (b)"),
    Line2D([0], [0], ls="none", marker="o", ms=4.6, mfc="white", mec=INK, mew=0.9, label="one clip's fps (b)"),
]
HANDLES = key_handles + style_handles + mark_handles
NCOL = 5


def add_legend(y_frac):
    return fig.legend(handles=HANDLES, loc="lower left", bbox_to_anchor=(XL, y_frac), ncol=NCOL, frameon=False,
                      fontsize=9, handlelength=1.8, handleheight=1.0, columnspacing=1.4, labelcolor=INK2,
                      borderaxespad=0)


def text_in(s, **kw):  # rendered text height in inches (independent of the figure height)
    t = fig.text(0, 0, s, **kw)
    bb = t.get_window_extent(renderer=renderer)
    t.remove()
    return bb.height / fig.dpi


fig = plt.figure(figsize=(FIG_W, 8.0), facecolor="white")
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
title_in, sub_in, foot_in = text_in(title_w, **TITLE_KW), text_in(sub_w, **SUB_KW), text_in(foot_w, **FOOT_KW)
leg = add_legend(0.0)
fig.canvas.draw()
leg_in = leg.get_window_extent(renderer=renderer).height / fig.dpi
leg.remove()
TOP_PAD = 0.22
SUB_GAP = 0.16
FIG_H = (TOP_PAD + title_in + SUB_GAP + sub_in + 0.66 + (Y_MAX - Y_MIN) * ROW_IN + 0.62 + leg_in + 0.12
         + foot_in + 0.12)
fig.set_size_inches(FIG_W, FIG_H)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
IN = 1.0 / FIG_H  # one inch in figure-height fraction

fig.text(XL, 1 - TOP_PAD * IN, title_w, ha="left", va="top", **TITLE_KW)
sub_top = TOP_PAD + title_in + SUB_GAP
fig.text(XL, 1 - sub_top * IN, sub_w, ha="left", va="top", **SUB_KW)
sub_bottom = 1 - (sub_top + sub_in) * IN
fig.text(XL, 0.12 * IN, foot_w, ha="left", va="bottom", **FOOT_KW)
foot_top = (0.12 + foot_in) * IN
leg = add_legend(foot_top + 0.12 * IN)
leg_top = foot_top + (0.12 + leg_in) * IN

AX_B = leg_top + 0.62 * IN
AX_T = sub_bottom - 0.66 * IN
AXA_X, AXA_W = 0.195, 0.495
AXB_X, AXB_W = 0.735, 0.235
axA = fig.add_axes([AXA_X, AX_B, AXA_W, AX_T - AX_B])
axB = fig.add_axes([AXB_X, AX_B, AXB_W, AX_T - AX_B], sharey=axA)
for ax in (axA, axB):
    ax.set_facecolor("none")  # transparent, so the band behind our operating modes shows through
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, length=3)
    ax.tick_params(axis="y", left=False, labelleft=False)
axA.set_ylim(Y_MIN, Y_MAX)


def yfig(yd):
    return fig.transFigure.inverted().transform(axA.transData.transform((0, yd)))[1]


for y_hi, y_lo in bands:
    yh, yl = yfig(y_hi), yfig(y_lo)
    fig.add_artist(Rectangle((XL - 0.006, yl), AXB_X + AXB_W + 0.012 - XL, yh - yl, transform=fig.transFigure,
                             fc=BAND, ec="none", zorder=-5))

# ---- panel a: KDE ridgeline
all_rows = [r for g in groups for r in g[1]]


def kde(v, gx):  # sdv2_latency.gaussian_kde
    bw = v.std(ddof=1) * v.size ** (-1 / 5)
    z = (gx[:, None] - v[None, :]) / bw
    return np.exp(-0.5 * z * z).sum(axis=1) / (v.size * bw * math.sqrt(2 * math.pi))


def kde_end(v):  # right end of the drawn curve (density > 0.1 % of the row's peak, as drawn below)
    bw = v.std(ddof=1) * v.size ** (-1 / 5)
    gx = np.linspace(0, v.max() + 6 * bw, 3000)
    d = kde(v, gx)
    return gx[np.nonzero(d > 0.001 * d.max())[0][-1]]


XMAX = math.ceil(max(kde_end(r["ms"]) for r in all_rows) / 500.0) * 500
grid = np.linspace(0, XMAX, 1000)  # as sdv2_latency

LS = {"orig": (0, (3, 2)), "opt": "solid", "alt": (0, (1, 1.5))}
for r in all_rows:
    kh = r["kde_h"]
    d = kde(r["ms"], grid)
    d = d / d.max() * kh
    b = r["base"]
    fc = lighten(r["colour"], 0.62) if r["variant"] in ("orig", "alt") else lighten(r["colour"], 0.30)
    axA.fill_between(grid, b, b + d, color=fc, lw=0, zorder=2)
    d_line = np.where(d > 0.001 * kh, b + d, np.nan)
    axA.plot(grid, d_line, color=r["colour"], lw=1.6 if r["kind"] != "alt" else 1.3, ls=LS[r["variant"]], zorder=3)
    axA.plot([0, XMAX], [b, b], color="#cfcfcf", lw=0.7, zorder=1)
    h50 = np.interp(r["p50"], grid, d)
    axA.plot([r["p50"]] * 2, [b, b + h50], color=INK, lw=1.8, zorder=4, solid_capstyle="butt")
    axA.plot([r["p99"]] * 2, [b, b + kh * 0.92], color=INK2, lw=1.3, ls=(0, (3, 2)), zorder=4)
    fs = 8.5 if r["kind"] != "alt" else 8
    axA.text(r["p99"] + 0.008 * XMAX, b + kh * 0.66, f"p99 {r['p99']:.0f}", fontsize=fs, color=INK2, ha="left",
             va="center", zorder=5)
    assert XMAX - r["p99"] > 0.11 * XMAX, r["label"]  # the label fits to the right of the tick
axA.axvline(BUDGET_MS, color=C_RT, lw=1.4, ls=(0, (6, 3)), zorder=6)
axA.text(BUDGET_MS, Y_MAX + 0.05, f"750{NB}ms = one 0.75{NB}s chunk of playback", color=C_RT, fontsize=9,
         fontweight="bold", ha="left", va="bottom", clip_on=False)
axA.set_xlim(0, XMAX)
axA.set_xticks(np.arange(0, XMAX + 1, 1000))
axA.set_xticklabels([f"{int(t):,}" for t in np.arange(0, XMAX + 1, 1000)])
axA.set_xlabel("per-frame action → frame latency, server_ms (ms; ↓ lower is better)", color=INK)
fig.text(XL, AX_T + 0.30 * IN, "a", fontsize=13, fontweight="bold", color=INK, ha="left", va="bottom")
fig.text(XL + 0.018, AX_T + 0.30 * IN,
         f"Distribution of per-frame latency ({FR_PER_CLIP} frames per clip, all clips of a config)",
         fontsize=11, color=INK, ha="left", va="bottom")

# group headers and row labels
yax = axA.get_yaxis_transform()
for kind, obj, yy, band in layout:
    if kind == "hdr":
        axA.annotate(obj, xy=(-(AXA_X - XL) / AXA_W, yy), xycoords=yax, ha="left", va="center", fontsize=9.5,
                     color=INK2 if band else MUTED, style="italic", fontweight="bold" if band else "normal",
                     annotation_clip=False, zorder=8,
                     bbox=dict(boxstyle="square,pad=0.12", fc=BAND if band else "white", ec="none"))
        continue
    r = obj
    kh = r["kde_h"]
    axA.add_patch(Rectangle((-0.020, r["base"]), 0.008, kh * 0.85, transform=yax, clip_on=False,
                            fc=r["colour"] if r["variant"] == "opt" else lighten(r["colour"], 0.45), lw=0))
    stats = f"p50 {r['p50']:.0f}{NB}ms · miss {fmt_miss(r['miss'])}"
    if r["clips"] != 6:
        stats += f" · n{NB}={NB}{r['clips']}"
    if r["kind"] == "alt":
        axA.annotate(r["name"], xy=(-0.032, r["mid"]), xycoords=yax, xytext=(0, 5.2), textcoords="offset points",
                     ha="right", va="center", fontsize=9, color=INK2, annotation_clip=False)
        axA.annotate(stats, xy=(-0.032, r["mid"]), xycoords=yax, xytext=(0, -5.2), textcoords="offset points",
                     ha="right", va="center", fontsize=8.3, color=MUTED, annotation_clip=False)
    else:
        axA.annotate(r["name"], xy=(-0.032, r["mid"]), xycoords=yax, xytext=(0, 6.5), textcoords="offset points",
                     ha="right", va="center", fontsize=10, fontweight="bold",
                     color=r["colour"] if r["kind"] == "mode" else INK, annotation_clip=False)
        axA.annotate(stats, xy=(-0.032, r["mid"]), xycoords=yax, xytext=(0, -6.5), textcoords="offset points",
                     ha="right", va="center", fontsize=9, color=INK2, annotation_clip=False)
fig.canvas.draw()
for yy in dividers:
    yf = yfig(yy)
    fig.add_artist(Line2D([XL, AXB_X + AXB_W], [yf, yf], transform=fig.transFigure, color="#c8c8c8", lw=0.8,
                          ls=(0, (4, 3))))

# ---- panel b: output fps bars on the same rows
FPS_MAX = 20.0
for r in all_rows:
    bh = 0.50 if r["kind"] != "alt" else 0.38
    if r["variant"] == "opt":
        axB.barh(r["mid"], r["fps"], height=bh, color=r["colour"], edgecolor="white", lw=0, zorder=2)
    else:
        axB.barh(r["mid"], r["fps"], height=bh, color=lighten(r["colour"], 0.62), edgecolor=r["colour"],
                 lw=1.2, ls=LS[r["variant"]], zorder=2)
    pc = [c["fps"] for c in r["per_clip"]]
    axB.scatter(pc, [r["mid"]] * len(pc), s=17 if r["kind"] != "alt" else 13, marker="o", fc="white", ec=INK,
                lw=0.9, zorder=4)
    axB.text(max(r["fps"], max(pc)) + 0.42, r["mid"], f"{r['fps']:.1f}", fontsize=10 if r["kind"] != "alt" else 9,
             fontweight="bold" if r["kind"] != "alt" else "normal", color=INK if r["kind"] != "alt" else INK2,
             ha="left", va="center", zorder=5)
axB.axvline(PLAYBACK_FPS, color=C_RT, lw=1.4, ls=(0, (6, 3)), zorder=6)
axB.text(PLAYBACK_FPS, Y_MAX + 0.05, f"16{NB}fps playback", color=C_RT, fontsize=9, fontweight="bold",
         ha="center", va="bottom", clip_on=False)
axB.set_xlim(0, FPS_MAX)
axB.set_xticks(np.arange(0, FPS_MAX + 0.1, 2))
axB.set_xlabel("output fps (frames/s; ↑ higher is better)", color=INK)
axB.spines["left"].set_visible(True)
fig.text(AXB_X - 0.02, AX_T + 0.30 * IN, "b", fontsize=13, fontweight="bold", color=INK, ha="left", va="bottom")
fig.text(AXB_X - 0.002, AX_T + 0.30 * IN, "Output fps (mean over clips)", fontsize=11, color=INK,
         ha="left", va="bottom")

# ---- layout check: no text box overlaps another (row labels, p99 labels, fps labels, headers)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
boxes = []
for ax in (axA, axB):
    for t in ax.texts:
        if t.get_text():
            boxes.append((t.get_text(), t.get_window_extent(renderer)))
for t in fig.texts:
    boxes.append((t.get_text()[:30], t.get_window_extent(renderer)))
bad = []
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        a, b = boxes[i][1], boxes[j][1]
        if a.x0 < b.x1 - 1 and b.x0 < a.x1 - 1 and a.y0 < b.y1 - 1 and b.y0 < a.y1 - 1:
            bad.append((boxes[i][0], boxes[j][0]))
print("text overlaps:", bad if bad else "none")

os.makedirs(OUT_FIG, exist_ok=True)
for ext in ("png", "svg"):
    fig.savefig(os.path.join(OUT_FIG, f"f13_latency.{ext}"), dpi=200, facecolor="white")
print("saved", os.path.join(OUT_FIG, "f13_latency.png"), "size", FIG_W, round(FIG_H, 2), "xmax", XMAX)

# ---------------------------------------------------------------- markdown table
COLS = [  # key, header, fmt, better
    ("fps", "output fps ↑", lambda x: f"{x:.1f}", "hi"),
    ("p50", "server_ms p50 ↓", lambda x: f"{x:.0f}", "lo"),
    ("p90", "p90 ↓", lambda x: f"{x:.0f}", "lo"),
    ("p99", "p99 ↓", lambda x: f"{x:.0f}", "lo"),
    ("miss", "miss > 750 ms ↓", fmt_miss, "lo"),
    ("first", "first frame after gen start ↓", lambda x: f"{x:.0f} ms", "lo"),
    ("gen", "gen s / 10 s clip ↓", lambda x: f"{x:.1f}", "lo"),
]
ranked = prof + modes  # baselines and operating modes; the other variants are listed but not ranked


def rank_marks(key, fmt, better):
    num = lambda r: float(fmt(r[key]).replace(" %", "").replace(" ms", ""))  # noqa: E731
    vals = sorted({num(r) for r in ranked}, reverse=(better == "hi"))
    out = {}
    for r in ranked:
        v = num(r)
        out[r["label"]] = "best" if v == vals[0] else ("second" if len(vals) > 1 and v == vals[1] else "")
    return out


marks = {k: rank_marks(k, f, b) for k, _, f, b in COLS}


def table_line(r, rank):
    cells = []
    for k, _, f, _ in COLS:
        s = f(r[k])
        m = marks[k].get(r["label"], "") if rank else ""
        cells.append(f"**{s}**" if m == "best" else (f"<u>{s}</u>" if m == "second" else s))
    name = r["tname"] if rank else f"*{r['tname']}*"
    return f"| {name} | {r['clips']} | " + " | ".join(cells) + " |"


hdr = "| config | clips | " + " | ".join(h for _, h, _, _ in COLS) + " |"
sep = "|---|---|" + "---|" * len(COLS)
lines = [
    "# f13 latency table (Chenfeng's sdv2_latency method; our port scripts/wm_latency.py)",
    "",
    f"Finding: {title.replace(NB, ' ').replace(chr(10), ' ')}.",
    "",
    f"Protocol (all rows): jar + sandcastle × seeds {'/'.join(SEEDS)}, 161 frames ({FR_PER_CLIP} output frames per "
    "clip), empty text prompt, one RTX PRO 6000 Blackwell GPU per run, streaming full-VAE decode on a side CUDA stream "
    "of the same GPU (decode competes with generation), prepare excluded.",
    "",
    f"- Profile rows (n = 6): scripts/sdvg_profile.sh. Original stack = bf16 eager (10/5 slide code, run fast0); "
    "optimized stack = FP8 W8A8 + SageAttention 2 + torch.compile + ring KV (run fast1); the two ran concurrently on "
    "two separate GPUs; no other generation jobs, a VLM judge server (vLLM, Qwen3-VL-32B) ran on another GPU; prepare "
    f"config means {prep_lo:.0f}–{prep_hi:.0f} s (both models, T5 on CPU).",
    "- Ours rows: scripts/timing_pass.sh, a later separate run on the optimized stack (2 lanes on 2 GPUs, no other "
    "generation jobs, VLM judges on other GPUs; facts §15), specwm.res_split_infer --stream-decode 1 --no-device: cloud 14B + streaming decode on the same "
    f"GPU, no device stage, frames decoded at the cloud resolution (realtime {size_str('ours704')}, quality "
    f"{size_str('ours832')}, 704 variants {size_str('c704_sep')}, 832 separate {size_str('u2_832_sep')}, 576 "
    f"{size_str('c576_pc1')}; loop.json lo_shape × 8).",
    f"- n = 5 for ours realtime, ours quality and 704 separate commit: castle s1234 excluded, because the warm-up "
    f"covered only jar's latent shape and that clip paid a cold torch.compile (gen {ex[0][0]:.1f} / {ex[1][0]:.1f} / "
    f"{ex[2][0]:.1f} s vs {ex[0][1]:.1f}–{ex[0][2]:.1f} / {ex[1][1]:.1f}–{ex[1][2]:.1f} / {ex[2][1]:.1f}–"
    f"{ex[2][2]:.1f} s for the kept clips; timing_pass/sandcastle_wave_impact/<arm>_s1234/loop.json). n = 6 for "
    "the other variants (per-task warm-up seed 1).",
    "",
    "Definitions (scripts/wm_latency.py): server_ms = frame ready in host memory − start of its chunk (action "
    "consumed). Output fps = (frames − 1) / (last ready − first ready) per clip (whole clip incl. the start, not "
    "steady fps; facts §13), mean over the clips. First frame = first ready − start of generation, mean over the "
    "clips. Miss = share of frames with server_ms > 750 ms (one chunk of playback). Gen s = loop.json gen_s "
    "(generation wall time, prepare excluded), mean over the clips.",
    "",
    "Bold = best, underlined = second best in each column among the baselines and our operating modes (on the "
    "displayed rounding). The other variants (italic, second table) are not operating modes and are not ranked.",
    "",
    hdr, sep,
]
lines += [table_line(r, True) for r in prof + modes]
lines += ["", "Other variants of ours (same timing pass; not operating modes):", "", hdr, sep]
lines += [table_line(r, False) for r in alts]
lines += [
    "",
    f"- 704 fused without partial commit is faster than the realtime mode but has {bj_fused:.0f} % boundary jumps "
    f"vs {bj_pc1:.0f} % with partial commit (facts §5; data/f02_boundary_counts.json).",
    f"- 576 fused + partial has the lowest latency measured (p50 {ff['p50']:.0f} ms) but runs at a lower resolution "
    f"({size_str('c576_pc1')}) and still misses 750 ms on {fmt_miss(ff['miss'])} of frames.",
    f"- Generation alone (quiet steady, decode excluded; facts §15): ours realtime {GEN_FPS['ours704']:.2f} fps, "
    f"quality {GEN_FPS['ours832']:.2f} fps, teacher-parity {GEN_FPS['parity']:.2f} fps, teacher "
    f"{GEN_FPS['teacher']:.2f} fps. For our two modes the output fps above ({f1(rt['fps'])} / {f1(qm['fps'])}, "
    "whole-clip means) are lower because the full-VAE decode shares the GPU (facts §13). Do not set the teacher's "
    f"output fps ({f1(o_t['fps'])}, a whole-clip mean, {EST['whole_over_steady']:.3f}× its steady rate) against its "
    f"{GEN_FPS['teacher']:.2f}: on the steady basis the teacher gives {EST['steady_fps']:.2f} fps with the streamed "
    f"decode vs {GEN_FPS['teacher']:.2f} fps generation only (data/e_steady_fps.json opt_target; facts §13 / §15). "
    "Teacher-parity has no output-fps row.",
    f"- SDVG scope: slower than the teacher on jar + sandcastle only. Here τ −2.2 takes {acc_mean['optimized']:.1f} "
    f"(optimized) / {acc_mean['original']:.1f} (original) of 13 chunks from the draft (hard prompts; facts §13); on "
    f"bench-43 it takes {B43['acc22']:.1f} and is faster than the teacher ({B43['sdvg22']:.1f} fps whole-clip / "
    f"{B43['sdvg22_steady']:.1f} steady vs {B43['teacher']:.1f}, under load; facts §2), at a quality cost "
    f"(Δ quality vs teacher {B43['dq22']}).",
    "- Teacher-parity mode (cw07_safe) has no latency row: spec_early has no streaming decode, so it records no "
    "per-frame ready times (needs --stream-decode support; facts §13). Matrix-Game 2.0 and Waypoint log no per-frame "
    "ready times either (whole-clip fps only, facts §15).",
    "",
    "Per clip, SDVG vs the teacher of the same stack (same task and seed): " + per_clip_txt.replace(NB, " "),
    "",
    "Sources: lingbot-world-v2/specwm_out/sdvg_profile/latency/<label>/summary.json (profile rows) and "
    "lingbot-world-v2/specwm_out/timing_pass/latency/<label>/summary.json (ours rows); both match "
    "paper/meeting_1012/facts.md §13 / §15. Bench-43 SDVG contrast: facts §2; teacher steady output fps: "
    "paper/meeting_1012/data/e_steady_fps.json. Generated by paper/meeting_1012/scripts/f13_latency.py.",
]
os.makedirs(os.path.dirname(OUT_TAB), exist_ok=True)
open(OUT_TAB, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("saved", OUT_TAB)
for r in all_rows:
    print(r["label"], r["clips"], {k: round(r[k], 2) for k in ("fps", "p50", "p90", "p99", "miss", "first", "gen")})
