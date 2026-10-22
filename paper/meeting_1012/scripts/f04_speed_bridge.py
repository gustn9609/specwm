#!/usr/bin/env python
"""f04_speed_bridge: where the speed-up comes from (cloud seconds per 0.75 s chunk, QUIET, steady).

Bridge / waterfall from the 14B teacher (today's inference stack, 4 steps) to ours 704 (realtime mode).
Bars = quiet steady s/chunk of the 10/12 timing pass (facts.md section 15; corrected 10/12):
    teacher 2.939 -> 14B 2-step 832 separate 1.504 -> ours quality mode (832 fused + partial) 1.232
    -> ours realtime mode (704 fused + partial) 0.763 s/chunk
The teacher's older "2.45 s/chunk = 4.9 fps quiet" (facts.md sections 3 and 5) is a WHOLE-CLIP average
(31.2 s / 13 chunks incl. the cheap early chunks before the KV window fills) and is no longer used.
Every number is read from the files below (nothing estimated):
  - paper/meeting_1012/facts.md      section 15 (quiet steady s/chunk + fps, the bars), 13 (original-stack
                                     target-only steady fps, the separate reference), 0 (systems ladder, blind
                                     test), 2 (bench-43 teacher steady under load), 3 (12-prompt x 3-seed quality),
                                     5 (boundary jumps 832 fused / fused + partial), 6 (one shared GPU + full VAE)
  - lingbot-world-v2/specwm_out/timing_pass/<task>/<arm>_s*/loop.json   raw cross-check of the four bars
  - paper/meeting_1012/data/e_steady_fps.json  original-code reference (orig_target steady_period_s;
                                     written by scripts/e_steady_fps.py from sdvg_profile/fast0)
  - paper/1012/bench43_summary.json  bench-43 absolute quality + paired quality deltas vs the teacher
The original-code reference is NOT a bridge step: it is target-only on the 10/5 stack (bf16 eager) with the
decode streamed on the same GPU (inside its s/chunk), while the bars exclude decode.
Re-run:  CUDA_VISIBLE_DEVICES="" /home/hk28965/specwm/venv/bin/python paper/meeting_1012/scripts/f04_speed_bridge.py
"""
import glob
import json
import os
import re
import statistics as st
import textwrap

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

ROOT = "/home/hk28965/specwm"
FACTS = f"{ROOT}/paper/meeting_1012/facts.md"
BENCH43 = f"{ROOT}/paper/1012/bench43_summary.json"
TPASS = f"{ROOT}/lingbot-world-v2/specwm_out/timing_pass"
E_STEADY = f"{ROOT}/paper/meeting_1012/data/e_steady_fps.json"
OUT = f"{ROOT}/paper/meeting_1012/figs/f04_speed_bridge"

FRAMES_PER_CHUNK = 12  # 3 latent frames = 12 video frames = 0.75 s at 16 fps (facts.md section 0)
PLAYBACK_FPS = 16


# ----------------------------------------------------------------------------- parsing helpers
def _clean(cell):
    cell = re.sub(r"</?u>", "", cell)
    return cell.replace("**", "").replace("*", "").strip()


def md_row(path, first_cell, section=None, pred=None):
    """Cleaned cells of the first markdown table row whose first cell == first_cell.
    section: only search after the first line that starts with this text."""
    with open(path) as f:
        lines = f.read().splitlines()
    start = 0
    if section is not None:
        start = next(i for i, l in enumerate(lines) if l.startswith(section))
    for line in lines[start:]:
        if not line.startswith("|"):
            continue
        cells = [_clean(c) for c in line.strip().strip("|").split("|")]
        if cells[0] == first_cell and (pred is None or pred(cells)):
            return cells
    raise KeyError(f"row '{first_cell}' not found in {path} (section {section})")


def num(cell):
    """First number in a cell ('7.53 ± 0.14' -> 7.53, '≈ 29' -> 29, '7 %' -> 7)."""
    m = re.search(r"-?\d+(?:\.\d+)?", cell.replace("−", "-"))
    return float(m.group(0))


def facts_regex(pattern):
    with open(FACTS) as f:
        m = re.search(pattern, f.read())
    if m is None:
        raise KeyError(pattern)
    return m


# ----------------------------------------------------------------------------- data
def tpass_steady(arm, key, skip=()):
    """raw quiet steady s/chunk (facts.md section 15 rules): spec_early arms = median over clips of
    median(diff(chunk_start)[5:]); res_split arms = median over clips of median(cloud_chunk_s[6:]).
    Seed 1 (per-task warm-up) and *_warm are excluded; `skip` = extra clips excluded (cold compile)."""
    per = []
    for f in sorted(glob.glob(f"{TPASS}/*/{arm}_s[0-9]*/loop.json")):
        clip = f.split("/")[-3] + "/" + f.split("/")[-2]
        if f.split("/")[-2].endswith("_s1") or clip in skip:
            continue
        r = json.load(open(f))
        v = np.diff(r["chunk_start"])[5:] if key == "chunk_start" else np.asarray(r["cloud_chunk_s"])[6:]
        per.append(float(np.median(v)))
    assert per, arm
    return st.median(per), len(per)


def load():
    d = {}
    # section 15 table (quiet timing pass): the four bars
    sec15 = "## 15."
    rows15 = {"T": "teacher 14B 4-step (uniform4)", "L2": "14B 2-step 832, separate commit",
              "O832": "ours quality mode (832 fused + partial)", "O704": "ours realtime mode (704 fused + partial)"}
    for k, key in rows15.items():
        c = md_row(FACTS, key, section=sec15)
        d[f"{k}_spc"], d[f"{k}_fps_s"], d[f"{k}_x"] = num(c[2]), c[3], c[4]
    # raw cross-check against lingbot-world-v2/specwm_out/timing_pass (fails loudly on a mismatch)
    castle_cold = ("sandcastle_wave_impact/ours832_s1234", "sandcastle_wave_impact/ours704_s1234")
    raw = {"T": tpass_steady("uniform4", "chunk_start"), "L2": tpass_steady("u2_832_sep", "cloud"),
           "O832": tpass_steady("ours832", "cloud", castle_cold), "O704": tpass_steady("ours704", "cloud", castle_cold)}
    for k, (v, n) in raw.items():
        assert abs(v - d[f"{k}_spc"]) < 0.0015, (k, v, d[f"{k}_spc"])
        d[f"{k}_n"] = n
    m = facts_regex(r"teacher's often-quoted '4\.9 fps quiet' is a whole-clip average \(gen ([\d.]+) s / (\d+) chunks")
    d["old_gen"], d["old_chunks"] = float(m.group(1)), int(m.group(2))
    t5 = md_row(FACTS, "teacher 832 (s1234)", section="## 5.")
    d["old_spc"], d["old_fps"] = num(t5[1]), num(t5[2])          # 2.45 / 4.9: named in the footer only
    m = facts_regex(r"teacher runs at [\d.]+ fps quiet / ([\d.]+) under load")
    d["teacher_load_fps"] = float(m.group(1))
    m = facts_regex(r"under-load ratios \(([\d.]+)× / ([\d.]+)× / ([\d.]+)×\) and quiet steady ratios")
    d["load_ratios"] = m.groups()

    # section 5: boundary jumps 832 fused alone vs fused + partial (seed 1234)
    sec5 = "## 5."
    d["f832_jump"] = num(md_row(FACTS, "832 fused (s1234)", section=sec5)[4])
    d["o832_jump"] = num(md_row(FACTS, "832 fused + partial 1 (s1234)", section=sec5)[4])

    # original-code reference (facts section 13 'Steady output fps'; data/e_steady_fps.json)
    e = json.load(open(E_STEADY))
    d["orig_spc"], d["orig_fps"] = e["orig_target"]["steady_period_s"], e["orig_target"]["steady_fps"]
    d["opt_spc"], d["orig_n"] = e["opt_target"]["steady_period_s"], e["orig_target"]["clips"]
    m = facts_regex(r"Original: draft [\d.]+, target ([\d.]+),")
    assert f"{d['orig_fps']:.2f}" == m.group(1), (d["orig_fps"], m.group(1))

    # section 3 table (12 prompts x 3 seeds, 36 clips): quality
    sec3 = "## 3."
    d["q12_teacher"] = num(md_row(FACTS, "14B 4-step teacher", section=sec3)[3])
    d["q12_two"] = num(md_row(FACTS, "14B 2-step 832 (separate commit)", section=sec3)[3])
    d["q12_o704"] = num(md_row(FACTS, "704 fused + partial 1", section=sec3)[3])

    # one shared GPU, cloud + full VAE (section 6; composed from the under-load cloud time)
    m = facts_regex(r"on one shared GPU: cloud \+ VAE ([\d.]+) s → ([\d.]+) fps")
    d["shared_spc"], d["shared_fps"] = float(m.group(1)), float(m.group(2))

    # systems-stack ladder + blind test (section 0): context only
    m = facts_regex(r"Measured ladder \(([^)]*)\): S0 original ([\d.]+) s .*?S3 \+ FP8 ([\d.]+) s "
                    r"\(4-step up to here\) → S4 2 steps ([\d.]+) s .*?S7 \+ SageAttention ([\d.]+) s")
    d["ladder_meta"] = m.group(1)
    d["S0"], d["S3"], d["S4"], d["S7"] = (float(m.group(i)) for i in range(2, 6))
    m = facts_regex(r"Blind test \((\d+) raters, (\d+) vs (\d+) clips\): S7 not worse than original "
                    r"\(severe-clean ([\d/]+) vs ([\d/]+); quality ([\d.]+) vs ([\d.]+)\)")
    d["blind"] = dict(raters=int(m.group(1)), n=int(m.group(2)), sc_s7=m.group(4), sc_orig=m.group(5),
                      q_s7=m.group(6), q_orig=m.group(7))

    # bench-43: absolute quality + paired deltas vs the teacher
    with open(BENCH43) as f:
        b = json.load(f)
    d["q43_teacher"], d["n43"] = b["teacher"]["quality"], b["teacher"]["clips"]
    for k in ("ours832", "ours704"):
        d[f"q43_{k}"] = b[k]["quality"]
        d[f"dq43_{k}"], d[f"dq43se_{k}"], d[f"n43_{k}"] = b[k]["d_quality"], b[k]["d_quality_se"], b[k]["paired_n"]
    return d


# ----------------------------------------------------------------------------- style
C_TEACHER, C_OURS832, C_OURS704 = "#1f1f1f", "#7b4bd1", "#d81b60"
C_OTHER = "#9e9e9e"       # intermediate total (2-step 832): "anything else greys"
C_DOWN = "#43a047"        # lever that removes time (green)
C_UP = "#e57373"          # lever that adds time back (light red)
INK, INK2, MUTED = "#1f1f1f", "#4a4a4a", "#7a7a7a"
C_RT = "#c62828"          # real-time reference line

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.edgecolor": "#9e9e9e", "axes.linewidth": 0.8,
})


def mfmt(v, nd=2, sign=True):
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s.replace("-", "−")


def main():
    d = load()
    rt_spc = FRAMES_PER_CHUNK / PLAYBACK_FPS   # 0.75 s
    T, L2, O832, O704 = d["T_spc"], d["L2_spc"], d["O832_spc"], d["O704_spc"]
    speedup = T / O704

    # bar list: (kind, tick label, lo, hi, colour)
    bars = [
        ("total", "Teacher 14B\n4 steps, 832", 0, T, C_TEACHER),
        ("down", "4 → 2\nsteps", L2, T, C_DOWN),
        ("total", "14B 2 steps\n832, separate\ncommit", 0, L2, C_OTHER),
        ("down", "fused +\npartial\ncommit", O832, L2, C_DOWN),
        ("total", "Ours 832\n(quality\nmode)", 0, O832, C_OURS832),
        ("down", "832 → 704\ncloud res.", O704, O832, C_DOWN),
        ("total", "Ours 704\n(realtime\nmode)", 0, O704, C_OURS704),
    ]
    ends = [T, L2, L2, O832, O832, O704, O704]   # level each bar ends on (connector start)

    fig = plt.figure(figsize=(12, 7.8))
    ax = fig.add_axes([0.075, 0.385, 0.785, 0.49])
    W = 0.62
    ymax = 3.95
    XREF = -1.45                                   # original-code reference, left of a divider

    for i, (kind, _, lo, hi, col) in enumerate(bars):
        ax.add_patch(Rectangle((i - W / 2, lo), W, hi - lo, facecolor=col, edgecolor="none", zorder=3))
        if i < len(bars) - 1:
            ax.plot([i + W / 2, i + 1 - W / 2], [ends[i], ends[i]], color="#9e9e9e", lw=0.9,
                    ls=(0, (2, 2)), zorder=2)

    # ---- original-code reference: off-scale hatched bar with a break, separated by a divider
    top_ref = 3.05
    ax.add_patch(Rectangle((XREF - W / 2, 0), W, top_ref, facecolor="white", edgecolor=C_TEACHER,
                           hatch="////", lw=1.0, zorder=3))
    for yb_ in (2.62, 2.70):                       # break marks (bar continues off scale)
        ax.plot([XREF - W / 2 - 0.06, XREF + W / 2 + 0.06], [yb_ - 0.04, yb_ + 0.04], color="white", lw=5,
                zorder=4, solid_capstyle="butt")
        ax.plot([XREF - W / 2 - 0.06, XREF + W / 2 + 0.06], [yb_ - 0.04, yb_ + 0.04], color=C_TEACHER, lw=0.9,
                zorder=5)
    ax.text(XREF, top_ref + 0.06, f"{d['orig_spc']:.2f} s · {d['orig_fps']:.2f} fps\n(off scale)", ha="center",
            va="bottom", fontsize=9.5, color=INK, fontweight="bold", zorder=6)
    ax.axvline(-0.8, color="#bdbdbd", lw=0.9, ls=(0, (4, 3)), zorder=1)

    # real-time reference line
    ax.axhline(rt_spc, color=C_RT, lw=1.4, ls=(0, (6, 3)), zorder=4)
    ax.text(1.008, rt_spc, f"real time\n{rt_spc:.2f} s / chunk\n= {PLAYBACK_FPS} fps playback",
            transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=9.5, color=C_RT,
            fontweight="bold", clip_on=False)

    ann = dict(ha="center", fontsize=9.5, zorder=6)

    def spc_txt(spc):
        return f"{spc:.3f} s" if spc < 1 else f"{spc:.2f} s"

    def total_label(i, spc, fps_s, extra=None, sep=" · "):
        ax.text(i, spc + 0.06, f"{spc_txt(spc)}{sep}{fps_s} fps", va="bottom", color=INK,
                fontweight="bold", **{**ann, "fontsize": 10.5})
        if extra:
            ax.text(i, spc + 0.26, extra, va="bottom", color=MUTED, **ann)

    total_label(0, T, d["T_fps_s"], "quiet GPU, steady")
    total_label(2, L2, d["L2_fps_s"])
    total_label(4, O832, d["O832_fps_s"])
    total_label(6, O704, d["O704_fps_s"], sep="\n")   # two lines: clears the 832 → 704 lever label

    # levers: delta seconds + factor (time before / time after)
    def lever_label(i, before, after, inside, extra=None):
        delta = after - before
        txt = f"{mfmt(delta)} s\n({before / after:.2f}×)"
        if inside:
            ax.text(i, (before + after) / 2, txt, va="center", color="white", fontweight="bold", **ann)
        else:
            top = max(before, after)
            ax.text(i, top + 0.06, txt, va="bottom", color=INK, fontweight="bold", **ann)
            if extra:
                ax.text(i, top + 0.40, extra, va="bottom", color=INK2, **ann)

    lever_label(1, T, L2, True)
    lever_label(3, L2, O832, False,
                f"jumps: fused alone {d['f832_jump']:.0f} %,\n+ partial {d['o832_jump']:.0f} % (§5)")
    lever_label(5, O832, O704, True)

    # ours 704 is just short of real time: say so explicitly
    over = (O704 / rt_spc - 1) * 100
    ax.annotate(f"{O704:.3f} s: {over:.1f} % over the {rt_spc:.2f} s budget\n"
                f"({d['O704_fps_s']} < {PLAYBACK_FPS} fps; cloud only, quiet).\n"
                f"One shared GPU + full VAE (§6,\nunder load): {d['shared_spc']:.2f} s → {d['shared_fps']:.1f} fps",
                xy=(6.0, O704 + 0.42), xytext=(6.0, 1.42), ha="center", va="bottom", fontsize=9,
                color=C_OURS704, zorder=6,
                arrowprops=dict(arrowstyle="-", color=C_OURS704, lw=0.8, shrinkA=2, shrinkB=0))

    # context box: systems stack (not bridged, different measure) + correctly attributed blind test
    bl = d["blind"]
    ctx = (
        "Context, not a bar: every bar already runs the systems stack (FP8, SageAttention 2, "
        f"torch.compile, ring KV); its own ladder is e2e per clip (§0, {d['ladder_meta'].split(',')[0]}): "
        f"{d['S0']:.1f} → {d['S3']:.1f} s (4 steps) → {d['S7']:.1f} s (2 steps). Blind test of S7 vs the "
        f"original: not worse ({bl['raters']} raters, {bl['n']} vs {bl['n']} clips; quality {bl['q_s7']} vs "
        f"{bl['q_orig']}); it does not test the systems stack alone."
    )
    ctx_txt = textwrap.fill(ctx, width=84)
    ax.text(1.55, 3.45, ctx_txt, ha="left", va="top", fontsize=9, color=INK2, zorder=7, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7f7f7", edgecolor="#d0d0d0", lw=0.8))

    # overall bracket (bridge bars only)
    yb = 3.66
    x0, x1 = 0 - W / 2, 6 + W / 2
    ax.plot([x0, x0, x1, x1], [yb - 0.07, yb, yb, yb - 0.07], color=INK2, lw=1.0, zorder=5, clip_on=False)
    ax.text((x0 + x1) / 2, yb + 0.04,
            f"{speedup:.2f}× less cloud time than the teacher = product of the three lever factors "
            "(each shown rounded to 2 decimals)",
            ha="center", va="bottom", fontsize=10, color=INK, zorder=6)

    # axes cosmetics
    ax.set_xlim(XREF - 0.55, len(bars) - 0.25)
    ax.set_ylim(0, ymax)
    ax.set_xticks([XREF] + list(range(len(bars))))
    ax.set_xticklabels(["Original code\n(10/5 stack)\nreference"] + [b[1] for b in bars], fontsize=10, color=INK)
    ax.get_xticklabels()[0].set_color(INK2)
    ax.get_xticklabels()[0].set_fontstyle("italic")
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    ax.set_ylabel("cloud seconds per 0.75 s chunk  (↓ lower is better)", color=INK)
    ax.grid(axis="y", color="#e6e6e6", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---------------------------------------------------------------- quality strip (two separate benchmarks)
    tr = blended_transform_factory(ax.transData, fig.transFigure)
    xl = 0.075
    rows = [
        (0.222, f"VLM quality ↑ (1–10)  ·  12 prompts × 3 seeds = 36 clips  ·  Δ vs teacher, unpaired",
         {0: f"{d['q12_teacher']:.2f} (ref.)",
          2: f"{d['q12_two']:.2f} ({mfmt(d['q12_two'] - d['q12_teacher'])})",
          4: "not run on 3 seeds",
          6: f"{d['q12_o704']:.2f} ({mfmt(d['q12_o704'] - d['q12_teacher'])})"}),
        (0.152, f"VLM quality ↑ (1–10)  ·  bench-43 = {d['n43']} clips, a different prompt set  ·  "
                "Δ vs teacher, paired ± SE  ·  not comparable with the row above",
         {0: f"{d['q43_teacher']:.2f} (ref.)",
          2: "not run",
          4: f"{d['q43_ours832']:.2f} ({mfmt(d['dq43_ours832'])} ± {d['dq43se_ours832']:.2f})",
          6: f"{d['q43_ours704']:.2f} ({mfmt(d['dq43_ours704'])} ± {d['dq43se_ours704']:.2f})"}),
    ]
    fig.add_artist(plt.Line2D([xl, 0.86], [0.292, 0.292], color="#d0d0d0", lw=0.8))
    for y, header, cells in rows:
        fig.text(xl, y + 0.030, header, fontsize=9.5, color=INK2, ha="left", va="bottom", style="italic")
        for i, s in cells.items():
            missing = s.startswith("not run")
            ax.text(i, y, s, transform=tr, ha="center", va="bottom", fontsize=10,
                    color=MUTED if missing else INK, fontweight="normal" if missing else "bold")

    # ---------------------------------------------------------------- title, subtitle, footer
    fig.text(xl, 0.968,
             f"Ours 704 cuts cloud time per chunk {d['O704_x']} vs the 14B teacher: "
             f"{T:.2f} → {O704:.3f} s ({d['T_fps_s']} → {d['O704_fps_s']} fps, quiet)",
             fontsize=13, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(xl, 0.932,
             "Green = cloud time a lever removes; bars = totals; × = time before / after. "
             f"Quiet steady speed-ups: quality mode {d['O832_x']}, realtime mode {d['O704_x']}.",
             fontsize=10, color=INK2, ha="left", va="top")

    lr = d["load_ratios"]
    footer_parts = [
        "Bars: facts.md §15 quiet timing pass, re-checked from lingbot-world-v2/specwm_out/timing_pass/*/<arm>_s*/loop.json; "
        "steady s per chunk, generation only (no decode);",
        "one RTX PRO 6000 Blackwell per lane, no other generation jobs; jar + sandcastle × seeds 1234/4321/9999, seed-1 "
        f"warm-up excluded (n = {d['T_n']}; ours n = {d['O704_n']}: castle s1234 paid a cold compile).",
        "Teacher = per-clip median of diff(chunk_start)[5:]; 2-step 832 and ours = per-clip median of cloud_chunk_s[6:]; "
        "median over clips; fps = 12 / s per chunk.",
        f"Older teacher “{d['old_spc']:.2f} s · {d['old_fps']:.1f} fps quiet” = whole-clip average ({d['old_gen']:.1f} s / "
        f"{d['old_chunks']} chunks incl. cheap early chunks). Bench-43 under load: teacher {d['teacher_load_fps']:.1f} fps "
        f"steady; ratios {lr[0]}× / {lr[1]}× / {lr[2]}× (≈ 5 % apart).",
        f"Original-code reference: 10/5 stack (bf16 eager), target only, jar + sandcastle, n = {d['orig_n']} (facts §13, "
        "data/e_steady_fps.json); its steady s/chunk includes the decode streamed on the same GPU,",
        f"so it is not a bridge step (optimized stack, same protocol: {d['opt_spc']:.2f} s). Quality: facts §3, "
        "paper/1012/bench43_summary.json; judge Qwen3-VL-32B, 8 frames. Jumps: §5, seed 1234.",
    ]
    footer = "\n".join(footer_parts)
    fig.text(xl - 0.06, 0.012, footer, fontsize=8.3, color=MUTED, ha="left", va="bottom", linespacing=1.4)

    # ---------------------------------------------------------------- layout check (texts inside canvas, no overlaps)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    fb = fig.bbox
    tx = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    bbs = [(t.get_text()[:50], t.get_window_extent(rend)) for t in tx]
    for s_, b_ in bbs:
        if b_.x0 < 2 or b_.y0 < 2 or b_.x1 > fb.width - 2 or b_.y1 > fb.height - 2:
            print("WARN outside canvas:", repr(s_))
    for i in range(len(bbs)):
        for j in range(i + 1, len(bbs)):
            a, b_ = bbs[i][1], bbs[j][1]
            if min(a.x1, b_.x1) - max(a.x0, b_.x0) > 2 and min(a.y1, b_.y1) - max(a.y0, b_.y0) > 2:
                print("WARN text overlap:", repr(bbs[i][0]), "<->", repr(bbs[j][0]))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT + ".png", dpi=200, facecolor="white")
    fig.savefig(OUT + ".svg", facecolor="white")
    levels = [T, L2, O832, O704]
    print("levels s/chunk:", [round(x, 4) for x in levels], "n:", d["T_n"], d["L2_n"], d["O832_n"], d["O704_n"])
    print("lever factors:", [round(a / b, 4) for a, b in zip(levels[:-1], levels[1:])])
    print("speed-up teacher -> ours704:", round(speedup, 4), " over RT budget %:", round(over, 2))
    print("orig ref s/chunk:", d["orig_spc"], d["orig_fps"], " opt:", d["opt_spc"])
    print("q12 teacher/2step/704:", d["q12_teacher"], d["q12_two"], d["q12_o704"])
    print("q43 teacher/832/704:", d["q43_teacher"], d["q43_ours832"], d["q43_ours704"])
    print("wrote", OUT + ".png", OUT + ".svg")


if __name__ == "__main__":
    main()
