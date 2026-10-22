"""f06_paired_deltas: forest plot of paired differences vs the teacher on bench-43.

Each arm minus the teacher (14B, 4-step) on the same prompt and seed, 86 pairs per arm,
mean +/- 1.96*SE, two panels (delta VLM quality up / delta VLM drift down) plus a narrow
fps panel (one GPU, under load). Arms sorted by delta quality.
Our three operating modes (names used across the meeting package, facts.md sections 1 and 12):
  ours: teacher-parity mode = cw07_safe (chunk-wise adaptive 0.7 + safe commit)
  ours: quality mode        = ours832   (832, 2-step, fused + partial commit)
  ours: realtime mode       = ours704   (~704x384, 2-step, fused + partial commit)

Data:
- /home/hk28965/specwm/paper/1012/bench43_summary.json (written by scripts/bench43_report.py --json;
  fields d_quality, d_quality_se, d_drift, d_drift_se, paired_n, fps, accepted; cw07_safe added 10/12,
  facts.md section 12). SE there =
  std(ddof=1)/sqrt(n) of the per-pair difference.
- per-clip generator logs lingbot-world-v2/specwm_out/bench43/<task>/<arm>_s*/loop.json
  (gen_s = generation seconds per clip, accepted = SDVG chunks taken from the draft,
  mean_steps = denoising steps per chunk for the chunk-wise arm).
- Timing caveats that have no raw file of their own (ours 704 on one shared GPU with full VAE / TAEHV)
  are quoted from paper/meeting_1012/facts.md section 6. The quiet timing pass (teacher 4.08 fps quiet
  steady; speed-ups of the three modes) is read at run time from the facts.md section 15 table. The
  teacher's older "4.9 fps quiet" is a whole-clip average (facts.md section 15 correction), not used.

How fps is defined per arm (scripts/bench43_report.py chunk_s): teacher / student = median chunk
interval after chunk 5 (also the chunk-wise teacher-parity mode); quality / realtime mode = median
cloud_chunk_s after chunk 6; SDVG = gen_s / 13 (whole clip,
incl. the forced chunk 0); Matrix-Game 2.0 = gen_fps and Waypoint = output_fps from their own
timing.json (whole run, own decoder included). LingBot arms exclude the final VAE decode
(SDVG's verify step decodes draft chunks for ImageReward; that cost is inside gen_s). Quality / realtime
mode decode on the device in parallel (loop.json decode_s 0); the teacher-parity mode is a single-GPU
14B run that decodes after generation like the teacher (loop.json decode_s, e.g. 8.59 vs teacher 8.5 s
on balloon_block_cut s1234), so leaving the decode out is like-for-like with the teacher.

The title's "only" is scoped to the plotted arms: two chunk-wise arms in bench43_summary.json that are
not plotted (cw07_pc1, cw07_sep) are also faster than the teacher and within noise on both axes; they
are found from the JSON and named in the last subtitle line (facts.md section 12).

Run: CUDA_VISIBLE_DEVICES="" /home/hk28965/specwm/venv/bin/python \
     /home/hk28965/specwm/paper/meeting_1012/scripts/f06_paired_deltas.py
"""
import glob
import json
import os
import textwrap

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

ROOT = "/home/hk28965/specwm"
SRC = ROOT + "/paper/1012/bench43_summary.json"
BENCH = ROOT + "/lingbot-world-v2/specwm_out/bench43"
OUT = ROOT + "/paper/meeting_1012/figs"
FID = "f06_paired_deltas"
Z = 1.96
NCHUNK = 13
SUB_WRAP = 148  # subtitle wrap width (characters)

# fixed arm colours (shared across all meeting figures)
COL = {"teacher": "#1f1f1f", "student": "#e07b00", "sdvg07": "#8d6e63", "sdvg22": "#c49a6c",
       "mg2": "#2e7d32", "wp": "#00897b", "ours832": "#7b4bd1", "ours704": "#d81b60",
       "cw07_safe": "#1565c0"}  # teacher-parity mode: new colour (lighter blues reserved for other chunk-wise arms)
NAME = {"teacher": "teacher 14B", "student": "student 1.3B", "sdvg07": "SDVG τ −0.7", "sdvg22": "SDVG τ −2.2",
        "mg2": "Matrix-Game 2.0", "wp": "Waypoint-1.5-1B", "cw07_safe": "ours: teacher-parity mode",
        "ours832": "ours: quality mode", "ours704": "ours: realtime mode"}
MODES = ("cw07_safe", "ours832", "ours704")  # our three operating modes (bold row labels)
SHORT = {"cw07_safe": "teacher-parity", "ours832": "quality", "ours704": "realtime"}
# speed-up vs the teacher = ratio of the rounded under-load fps, as quoted in facts.md section 12 (1.74x / 2.5x / 4.1x)
SPEEDUP_ND = {"cw07_safe": 2, "ours832": 1, "ours704": 1}
# grey descriptor lines (facts.md sections 0, 1, 2, 9); SDVG lines are filled from the raw logs below
DESC = {"student": ["official causal-fast, 4-step"],
        "mg2": ["640×352, keyboard/mouse model"],
        "wp": ["1024×512, keyboard/mouse model"],
        "ours832": ["832, 14B 2-step, fused + partial"],
        "ours704": ["≈704×384, 14B 2-step, fused + partial"]}
ARMS = ["student", "sdvg07", "sdvg22", "mg2", "wp", "cw07_safe", "ours832", "ours704"]
# other chunk-wise arms in bench43_summary.json (facts.md section 12), not plotted; named in the subtitle
# when they also pass both noise tests, so the title's "only" stays scoped to the plotted arms
XNAME = {"cw07_pc1": "cw07_pc1 (0.7, fused + partial commit)",
         "cw07_sep": "cw07_sep (0.7, separate commit)",
         "cw05_pc1": "cw05_pc1 (0.5, fused + partial commit)"}
OWN_DECODER = {"mg2", "wp"}  # baselines whose fps includes their own decoder (facts.md section 2)
# per-clip generator logs (bench43/<task>/<pattern>/loop.json), same arm -> directory map as bench43_report.py
LOOP = {"teacher": "uniform4_s*", "student": "uniform4_1p3b_s*", "sdvg07": "ir_-0.7_s*", "sdvg22": "ir_-2.2_s*",
        "ours832": "pc1_832_s*", "ours704": "pc1_704_s*", "cw07_safe": "cw07_safe_s*"}

# timing caveats quoted from facts.md (no separate raw file): section 2 line 28, section 6, section 10
# (ours704 = realtime mode)
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
           "cw07_safe": "ours teacher-parity mode (chunk-wise 0.7, safe commit)",
           "ours832": "ours quality mode (832 fused + partial)",
           "ours704": "ours realtime mode (704 fused + partial)"}
    return {a: (rows[k][3], rows[k][4]) for a, k in key.items()}


QS = quiet_steady()
TEACHER_QUIET_FPS = QS["teacher"][0]  # 4.08 = steady quiet (facts.md section 15), NOT the whole-clip 4.9
OURS704_SHARED_VAE_FPS, OURS704_SHARED_TAEHV_FPS = 10.1, 15.5

INK, INK2, MUTED, GRID = "#1f1f1f", "#4a4a4a", "#7a7a7a", "#e6e6e6"


def sgn(v, nd):
    s = f"{abs(v):.{nd}f}"
    return ("+" if v >= 0 else "−") + s


def join(names):
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def loops(k):
    if k not in LOOP:
        return []
    return [json.load(open(f)) for f in sorted(glob.glob(f"{BENCH}/*/{LOOP[k]}/loop.json"))]


def gen_median(k):
    g = [r["gen_s"] for r in loops(k) if r.get("gen_s") is not None]
    return float(np.median(g)) if g else None


def check_overlaps(fig):
    """print any pair of visible texts whose boxes intersect, and any text outside the canvas"""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    ts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    bb = [(t, t.get_window_extent(r)) for t in ts]
    bad = 0
    for i, (t, b) in enumerate(bb):
        if b.x0 < 0 or b.y0 < 0 or b.x1 > W or b.y1 > H:
            print("OUTSIDE:", repr(t.get_text()[:50])); bad += 1
        for t2, b2 in bb[i + 1:]:
            if b.overlaps(b2):
                print("OVERLAP:", repr(t.get_text()[:40]), "<>", repr(t2.get_text()[:40])); bad += 1
    print("text overlap check:", "OK" if bad == 0 else f"{bad} problems")


def speedup(k, T, S):
    return f"{S[k]['fps'] / T['fps']:.{SPEEDUP_ND[k]}f}×"


def qspeed(k, T, S, fps=True):
    """speed vs the teacher, quiet steady ratio first (facts.md section 15, preferred for slides), then the
    bench-43 under-load ratio and fps; arms without a section-15 row get the under-load numbers only"""
    if k in QS:
        return (f"{QS[k][1]} the teacher's speed quiet ({speedup(k, T, S)} under load"
                + (f", {S[k]['fps']:.1f} vs {T['fps']:.1f} fps)" if fps else ")"))
    return f"{S[k]['fps']:.1f} vs {T['fps']:.1f} fps under load ({speedup(k, T, S)})"


def qname(k):
    """name in running text; our mode names are quoted so 'ours: ... mode' parses inside a sentence"""
    return f"“{NAME[k]}”" if k in MODES else NAME[k]


def join_arms(keys):
    """names in running text; all three of our modes collapse to one phrase"""
    keys = list(keys)
    if all(m in keys for m in MODES):
        rest = [qname(k) for k in keys if k not in MODES]
        return join(rest + ["all three of our modes"])
    return join(qname(k) for k in keys)


def main():
    S = json.load(open(SRC))
    T = S["teacher"]
    for k in ARMS:
        for f in ("d_quality", "d_quality_se", "d_drift", "d_drift_se", "fps", "paired_n"):
            assert S[k].get(f) is not None, (k, f)
    npairs = sorted({S[k]["paired_n"] for k in ARMS})

    # SDVG descriptors from the per-clip logs: mean chunks taken from the draft (chunk 0 is always forced
    # to the target, so at most 12 of 13), and how many clips take none
    for k in ("sdvg07", "sdvg22"):
        acc = np.array([r["accepted"] for r in loops(k)], float)
        assert len(acc) and abs(acc.mean() - S[k]["accepted"]) < 1e-9, (k, acc.mean(), S[k]["accepted"])
        d = [f"mean {acc.mean():.3g} / {NCHUNK} chunks from draft"]
        nz = int((acc == 0).sum())
        d.append(f"({nz}/{len(acc)} clips: none)" if nz else f"(median {np.median(acc):.0f})")
        DESC[k] = d
    # teacher-parity mode: mean denoising steps per chunk from its per-clip logs (facts.md section 12: 3.01)
    cw = [r for r in loops("cw07_safe")]
    assert len(cw) == S["cw07_safe"]["clips"], (len(cw), S["cw07_safe"]["clips"])
    cw_steps = float(np.mean([r["mean_steps"] for r in cw]))
    DESC["cw07_safe"] = ["14B chunk-wise adaptive 0.7,", f"mean {cw_steps:.2f} steps, safe commit"]

    order = sorted(ARMS, key=lambda k: S[k]["d_quality"], reverse=True)  # best Δ quality on top
    n = len(order)
    ypos = {k: n - 1 - i for i, k in enumerate(order)}

    qtie = lambda k: abs(S[k]["d_quality"]) < Z * S[k]["d_quality_se"]  # 95 % CI of the paired Δ includes 0
    dtie = lambda k: abs(S[k]["d_drift"]) < Z * S[k]["d_drift_se"]
    faster = [k for k in ARMS if S[k]["fps"] > T["fps"]]

    # ---- finding (computed, not hard-coded): arms faster than the teacher that are within noise on both axes
    both = [k for k in faster if qtie(k) and dtie(k)]
    # the same test on the bench-43 arms that are not plotted (the title's "only" covers the plotted arms)
    extra = [k for k in XNAME if k in S and S[k].get("d_quality") is not None
             and S[k]["fps"] > T["fps"] and qtie(k) and dtie(k)]
    extra.sort(key=lambda k: S[k]["fps"], reverse=True)
    if len(both) == 1:
        k = both[0]
        b = S[k]
        title = (f"Of the arms shown, only {qname(k)} is faster than the teacher and within noise on both axes:\n"
                 f"Δ quality {sgn(b['d_quality'], 2)} ± {b['d_quality_se']:.2f}, Δ drift {sgn(b['d_drift'], 1)} ± "
                 f"{b['d_drift_se']:.1f} (± SE, paired, {b['paired_n']} clips) at {qspeed(k, T, S, fps=False)}")
    else:
        best_q = max(faster, key=lambda k: S[k]["d_quality"])
        best_d = min(faster, key=lambda k: S[k]["d_drift"])
        title = (f"Among the arms faster than the teacher, closest on quality: {NAME[best_q]} "
                 f"({sgn(S[best_q]['d_quality'], 2)}); on drift: {NAME[best_d]} ({sgn(S[best_d]['d_drift'], 1)})")

    sub = [("Arm minus teacher on the same prompt and seed; marker = mean, bar = ± 1.96·SE of the paired difference; "
            f"fps under load (teacher {T['fps']:.1f}; {TEACHER_QUIET_FPS} quiet).")]
    # quality: arms whose 95 % CI includes 0; speed backed by per-clip generation time where the log has it
    tie = [k for k in order if qtie(k)]
    if tie:
        verb = "is" if len(tie) == 1 else "are"
        s = (f"On quality, of the arms shown only " + join(f"{qname(k)} ({sgn(S[k]['d_quality'], 2)})" for k in tie)
             + f" {verb} within noise of the teacher")
        slow = [k for k in tie if S[k]["fps"] <= T["fps"]]
        fast = [k for k in tie if S[k]["fps"] > T["fps"]]
        tg = gen_median("teacher")
        parts = []
        if slow:
            g = {k: gen_median(k) for k in slow}
            if tg is not None and all(v is not None for v in g.values()):  # per-clip generation time (sturdier)
                ev = (", ".join(f"{g[k]:.1f} s" for k in slow) + f" vs teacher {tg:.1f} s per clip, median")
            else:
                ev = ", ".join(f"{S[k]['fps']:.1f} fps" for k in slow) + f" vs teacher {T['fps']:.1f} fps"
            parts.append(f"{join(qname(k) for k in slow)} {'is' if len(slow) == 1 else 'are'} not faster ({ev})")
        if fast:
            parts.append(", ".join(f"{qname(k)} runs at {qspeed(k, T, S)}" for k in fast))
        # one fragment per clause so a wrapped line never breaks inside a number or parenthesis
        sub.append([s + (";" if parts else "."), ", while ".join(parts) + "."] if parts else [s + "."])
    # drift: arms whose 95 % CI includes 0, and the range of the rest
    dt = [k for k in order if dtie(k)]
    rest = [k for k in ARMS if k not in dt]
    if dt:
        s = (f"On drift, {join_arms(dt)} {'is' if len(dt) == 1 else 'are'} within noise "
             f"of the teacher (95 % CI includes 0)")
        if rest:
            lo, hi = min(S[k]["d_drift"] for k in rest), max(S[k]["d_drift"] for k in rest)
            worse = all(S[k]["d_drift"] > 0 for k in rest)
            s += (f"; the other {len(rest)} are {sgn(lo, 1)} to {sgn(hi, 1)} " + ("worse" if worse else "off"))
        sub.append(s + ".")
    # our three modes: speed vs quality trade
    ms = sorted(MODES, key=lambda k: S[k]["fps"])
    sub.append("Our modes trade quality for speed: " + ", ".join(
        f"{SHORT[k]} {QS[k][1]} (" + ("Δ quality " if i == 0 else "") + f"{sgn(S[k]['d_quality'], 2)})"
        for i, k in enumerate(ms))
               + " (quiet steady speed-ups, facts §15); bench-43 under load: "
               + " / ".join(f"{S[k]['fps']:.1f}" for k in ms) + " fps = "
               + " / ".join(speedup(k, T, S) for k in ms) + ".")
    # not plotted: other chunk-wise arms that also pass both tests; non-breaking spaces (NB) keep every
    # "value ± SE" and "x fps" group on one line when textwrap wraps the sentence
    if extra:
        NB = " "
        pm = lambda m, se, nd: f"{sgn(m, nd)}{NB}±{NB}{se:.{nd}f}"
        sub.append(f"Not shown: {len(extra)} more chunk-wise {'arm is' if len(extra) == 1 else 'arms are'} also "
                   "faster than the teacher and within noise on both axes (facts §12): " + "; ".join(
                       f"{XNAME[k]} {S[k]['fps']:.1f}{NB}fps, "
                       + (f"Δ{NB}quality{NB}" if i == 0 else "") + pm(S[k]['d_quality'], S[k]['d_quality_se'], 2) + ", "
                       + (f"Δ{NB}drift{NB}" if i == 0 else "") + pm(S[k]['d_drift'], S[k]['d_drift_se'], 1)
                       for i, k in enumerate(extra)) + ".")

    # ---- layout in inches from the top; rows a bit taller than before (0.494 in) so two adjacent
    #      three-line labels (SDVG tau -0.7 above the teacher-parity mode) keep a clear gap
    ROW_IN = 0.54
    W = 12.0
    sub_lines = [ln for x in sub for frag in ([x] if isinstance(x, str) else x) for ln in textwrap.wrap(frag, SUB_WRAP)]
    sub_txt = "\n".join(sub_lines)
    NSUB_LINES = len(sub_lines)
    n_title = title.count("\n") + 1
    title_y, line_h = 0.22, 10.5 * 1.45 / 72  # title top; subtitle line pitch
    sub_top = title_y + n_title * 13 * 1.35 / 72 + 0.14
    ax_top = sub_top + NSUB_LINES * line_h + 0.44
    ax_h = (n + 0.65) * ROW_IN
    ax_bot = ax_top + ax_h
    foot_y0, foot_dy = ax_bot + 0.92, 0.183

    foot = [
        (f"Data: bench-43 = 38 StEvo camera-WM tasks + 5 easy scenes × seeds 1234/4321 (161 frames); "
         f"{'/'.join(map(str, npairs))} paired clips per arm vs teacher 14B 4-step. Judge: Qwen3-VL-32B-Instruct,"),
        ("8 frames at 448 px; drift prompt AUROC 0.82 vs human labels, quality prompt not human-validated. "
         f"SDVG: chunk 0 is always forced to the 14B target (≤ {NCHUNK - 1} / {NCHUNK} chunks from draft)."),
        ("Teacher-parity mode (cw07_safe): a learned head on the 14B stops each chunk once ≥ 70 % of tokens are "
         "predicted converged (head time is in the fps);"),
        ("safe commit = fused + partial commit if the chunk stopped at step ≥ 3, full clean commit pass at step 1–2. "
         "Quality / realtime mode = 14B 2-step at 832 / ≈704×384."),
        ("Speed: one RTX PRO 6000 under load (7 concurrent jobs). Quiet timing pass (facts.md §15, jar + castle, steady): "
         f"teacher {TEACHER_QUIET_FPS} fps vs {T['fps']:.1f} here;"),
        (f"teacher-parity / quality / realtime {QS['cw07_safe'][1]} / {QS['ours832'][1]} / {QS['ours704'][1]} quiet "
         f"(vs {' / '.join(speedup(k, T, S) for k in MODES)} here). The teacher's older “4.9 fps quiet” was a whole-clip average."),
        ("fps = 12 / s per chunk: steady-state median for teacher / student / our modes; whole-clip mean for SDVG "
         "(incl. chunk 0); whole-run fps from own timing.json for MG2 / Waypoint."),
        ("Not like-for-like: LingBot arms (teacher, student, SDVG, our modes) exclude the final VAE decode (quality / "
         "realtime modes decode on the device in parallel;"),
        ("teacher-parity mode decodes after generation like the teacher; SDVG's verify-time decode is counted). "
         "† Matrix-Game 2.0 / Waypoint include their own decoders."),
        (f"Realtime mode on one shared GPU incl. decode: {OURS704_SHARED_VAE_FPS} fps with the full VAE, "
         f"{OURS704_SHARED_TAEHV_FPS} fps with TAEHV."),
        ("Source: paper/1012/bench43_summary.json (scripts/bench43_report.py), per-clip loop.json in "
         "lingbot-world-v2/specwm_out/bench43/; timing notes from facts.md §2, §6, §12, §15."),
    ]
    H = foot_y0 + (len(foot) - 1) * foot_dy + 0.26
    fy = lambda y: 1 - y / H  # inches from the top -> figure fraction

    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "axes.edgecolor": "#9a9a9a",
                         "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 10, "ytick.labelsize": 10})
    fig = plt.figure(figsize=(W, H), facecolor="white")
    top, bot = fy(ax_top), fy(ax_bot)
    axq = fig.add_axes([0.240, bot, 0.292, top - bot])
    axd = fig.add_axes([0.567, bot, 0.255, top - bot], sharey=axq)
    axf = fig.add_axes([0.853, bot, 0.127, top - bot], sharey=axq)

    def style(ax):
        ax.set_facecolor("white")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", colors=INK2, length=3)
        for k in order:  # faint row separators
            ax.axhline(ypos[k] - 0.5, color="#f2f2f2", lw=0.8, zorder=0)

    for ax in (axq, axd, axf):
        style(ax)
    ytop = n - 0.4 + 0.3
    axq.set_ylim(-0.6, n - 0.4 + 0.45)
    axq.set_yticks([])

    def forest(ax, fm, fse, nd, side):
        """side='left': value label left of the lower CI end (keeps it off the zero line); 'right': right of upper end"""
        ax.axvline(0, color=COL["teacher"], lw=1.6, zorder=2)
        for k in order:
            y, m, se = ypos[k], S[k][fm], S[k][fse]
            lo, hi = m - Z * se, m + Z * se
            ax.plot([lo, hi], [y, y], color=COL[k], lw=2.2, solid_capstyle="round", zorder=3)
            for e in (lo, hi):
                ax.plot([e, e], [y - 0.13, y + 0.13], color=COL[k], lw=1.6, zorder=3)
            ax.plot([m], [y], marker="o", ms=9, mfc=COL[k], mec="white", mew=1.6, zorder=4)
            ax.annotate(sgn(m, nd), xy=(lo if side == "left" else hi, y), xytext=(-5 if side == "left" else 5, 0),
                        textcoords="offset points", ha="right" if side == "left" else "left", va="center",
                        fontsize=9.5, color=INK2, zorder=5, fontweight="bold" if k in MODES else "normal")
        ax.text(0, ytop, "teacher = 0  " if side == "left" else "  teacher = 0",
                ha="right" if side == "left" else "left", va="center", fontsize=9, color=INK2)

    # ---- quality panel
    forest(axq, "d_quality", "d_quality_se", 2, "left")
    qlo = min(S[k]["d_quality"] - Z * S[k]["d_quality_se"] for k in ARMS)
    qhi = max(S[k]["d_quality"] + Z * S[k]["d_quality_se"] for k in ARMS)
    axq.set_xlim(qlo - 0.62, max(qhi, 0) + 0.25)
    axq.set_xticks(np.arange(np.ceil(qlo - 0.62), np.floor(max(qhi, 0) + 0.25) + 1))  # only in-view ticks
    axq.set_xlabel("Δ VLM quality vs teacher (1–10 scale)\n↑ higher is better", fontsize=11, color=INK)
    axq.set_title("Δ quality  (arm − teacher)", fontsize=11, color=INK, loc="left", pad=6)

    # ---- drift panel
    forest(axd, "d_drift", "d_drift_se", 1, "right")
    dlo = min(S[k]["d_drift"] - Z * S[k]["d_drift_se"] for k in ARMS)
    dhi = max(S[k]["d_drift"] + Z * S[k]["d_drift_se"] for k in ARMS)
    axd.set_xlim(min(dlo, 0) - 2.5, dhi + 6.5)
    axd.set_xlabel("Δ VLM drift vs teacher (0–100 scale)\n↓ lower is better", fontsize=11, color=INK)
    axd.set_title("Δ drift  (arm − teacher)", fontsize=11, color=INK, loc="left", pad=6)
    plt.setp(axd.get_yticklabels(), visible=False)

    # ---- fps panel (one GPU, under load); † = includes its own decoder
    fmax = max(S[k]["fps"] for k in ARMS)
    for k in order:
        y, f = ypos[k], S[k]["fps"]
        axf.barh(y, f, height=0.34, color=COL[k], zorder=3)
        lab = f"{f:.1f}" + ("†" if k in OWN_DECODER else "")
        # a bar that ends left of the teacher line gets its label right of the line, on a white box
        x0, off = (T["fps"], 6) if f < T["fps"] else (f, 4)
        axf.annotate(lab, xy=(x0, y), xytext=(off, 0), textcoords="offset points", ha="left", va="center",
                     fontsize=9.5, color=INK2, zorder=5, fontweight="bold" if k in MODES else "normal",
                     bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    axf.axvline(T["fps"], color=COL["teacher"], lw=1.2, ls=(0, (3, 2)), zorder=2)
    axf.text(T["fps"], ytop, f" teacher {T['fps']:.1f}", ha="left", va="center", fontsize=9, color=INK2)
    axf.set_xlim(0, fmax * 1.42)
    axf.set_xticks(np.arange(0, fmax * 1.42 + 1e-9, 10))  # every in-view tick (0 … 30)
    axf.set_xlabel("fps, under load\n↑ higher is better", fontsize=11, color=INK)
    axf.set_title("speed (under load)", fontsize=11, color=INK, loc="left", pad=6)
    plt.setp(axf.get_yticklabels(), visible=False)

    # ---- row labels (name + grey descriptor lines), left of the quality panel
    tr = blended_transform_factory(fig.transFigure, axq.transData)
    for k in order:
        y = ypos[k]
        lines = DESC[k]
        y0 = y + (0.15 if len(lines) == 1 else 0.33)
        fig.text(0.226, y0, NAME[k], transform=tr, ha="right", va="center", fontsize=10.5,
                 color=COL[k] if k in MODES else INK, fontweight="bold" if k in MODES else "normal")
        for j, s in enumerate(lines):
            fig.text(0.226, y0 - 0.35 - 0.32 * j, s, transform=tr, ha="right", va="center", fontsize=9, color=MUTED)

    # ---- title, subtitle, footer
    ttl = fig.text(0.012, fy(title_y), title, fontsize=13, color=INK, ha="left", va="top", fontweight="bold",
                   linespacing=1.35)
    rr = fig.canvas.get_renderer()
    for fs in (13, 12.5, 12, 11.5, 11):  # shrink only if a line would run off the canvas (same rule as f05)
        ttl.set_fontsize(fs)
        if ttl.get_window_extent(rr).x1 <= fig.bbox.width - 12:
            break
    else:
        print("[warn] title wider than the canvas even at 11 pt")
    fig.text(0.012, fy(sub_top), sub_txt, fontsize=10.5, color=INK2, ha="left", va="top", linespacing=1.45)
    for i, s in enumerate(foot):
        fig.text(0.012, fy(foot_y0 + i * foot_dy), s, fontsize=9, color="#8a8a8a", ha="left", va="center")

    check_overlaps(fig)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{FID}.png", dpi=200, facecolor="white")
    fig.savefig(f"{OUT}/{FID}.svg", facecolor="white")
    print(f"{OUT}/{FID}.png", f"({W} x {H:.2f} in)")
    print("order:", [(k, S[k]["d_quality"], S[k]["d_drift"], S[k]["fps"]) for k in order])
    print("per-clip gen median (s):", {k: gen_median(k) for k in LOOP})
    print("cw07_safe mean steps (loop.json):", round(cw_steps, 4))
    print("title:", title)
    print("sub:", *sub, sep="\n  ")


if __name__ == "__main__":
    main()
