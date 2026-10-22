#!/usr/bin/env python
"""f03_splits_map: status map of the cloud/edge splits and acceleration routes explored to date.

Re-run:  CUDA_VISIBLE_DEVICES="" /home/hk28965/specwm/venv/bin/python \
         /home/hk28965/specwm/paper/meeting_1012/scripts/f03_splits_map.py

Every number on the figure is read at run time from
  - paper/meeting_1012/facts.md          (sections 0, 1, 2, 3, 5, 6, 7, 8, 10, 15)
  - paper/1012/bench43_summary.json       (bench-43 fps / quality / drift / paired deltas per arm)
  - paper/1012/README.md sections L4, T   (verify + redraw table; Draft-then-Finish A/B)
The script fails loudly if a pattern is no longer found, so a stale figure cannot be produced silently.
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path("/home/hk28965/specwm")
MEET = ROOT / "paper/meeting_1012"
FACTS = (MEET / "facts.md").read_text()
README = (ROOT / "paper/1012/README.md").read_text()
B43 = json.loads((ROOT / "paper/1012/bench43_summary.json").read_text())
OUT = MEET / "figs"
FIG_ID = "f03_splits_map"


def grab(text, pattern, name, flags=re.S):
    m = re.search(pattern, text, flags)
    if m is None:
        raise SystemExit(f"[{FIG_ID}] pattern for '{name}' not found: {pattern}")
    return m.groups() if len(m.groups()) > 1 else m.group(1)


def neg(s):
    """ASCII minus -> typographic minus."""
    return s.replace("-", "−") if isinstance(s, str) else s


def sgn(x, nd=2):
    return f"{x:+.{nd}f}".replace("-", "−")


# ---------------------------------------------------------------- numbers
# facts §0: systems ladder (14B, e2e per clip)
n_sys = grab(FACTS, r"Measured ladder \([^,]+, (\d+) prompts, 14B", "systems ladder prompts")
s0, s3, s7 = grab(FACTS, r"S0 original ([\d.]+) s .*?S3 \+ FP8 ([\d.]+) s \(4-step up to here\).*?"
                         r"S7 \+ SageAttention ([\d.]+) s", "systems ladder")
# facts §1: 704 cloud s/chunk
cloud704 = grab(FACTS, r"Cloud ([\d.]+) s/chunk → [\d.]+ fps \(measured under load", "704 cloud s/chunk")
# facts §2: load condition of bench-43
n_load = grab(FACTS, r"UNDER LOAD \((\d+) jobs concurrently", "bench-43 load")
# facts §15: quiet timing pass (steady s/chunk, jar + castle). The teacher's older "4.9 fps quiet" (§3) is a
# whole-clip average incl. the cheap early chunks, so speed-ups are taken from §15 (facts §15 correction).
tq_s, tq_fps = grab(FACTS, r"\| teacher 14B 4-step \(uniform4\) \| [\d.]+ \| ([\d.]+) \| ([\d.]+) \| 1\.00× \|",
                    "teacher quiet steady (§15)")
cwq_steps, cwq_s, cwq_fps, cwq_x = grab(
    FACTS, r"\| ours teacher-parity mode \(chunk-wise 0\.7, safe commit\) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+×) \|",
    "teacher-parity quiet steady (§15)")
# facts §3: refiner cost (12 x 3 set)
ref_n, ref_d, ref_se = grab(FACTS, r"Refiner cost \(paired by clip, (\d+)\): 704 ([−\-\d.]+) ± ([\d.]+) quality",
                            "refiner cost")
# facts §5: boundary jumps at 704, separate vs fused
j704_sep = grab(FACTS, r"\| 704 separate \| [^|]+\| [^|]+\| [^|]+\| (\d+) % \|", "704 separate jumps")
j704_fused = grab(FACTS, r"\| 704 fused \| [^|]+\| [^|]+\| [^|]+\| (\d+) % \|", "704 fused jumps")
# facts §6: decoder timing / shared-GPU fps
vae704 = grab(FACTS, r"At 704: full Wan2\.1 VAE (\d+) ms/chunk", "VAE 704 ms/chunk")
shared_fps = grab(FACTS, r"on one shared GPU: cloud \+ VAE [\d.]+ s → ([\d.]+) fps", "shared-GPU fps")
# facts §7: draft-vs-target gap, physics verifier cost + prompt set
gap_sigma, gap_eps = grab(FACTS, r"draft chunks ([\d.]+) σ from the 14B output, (\d+) % of tokens within ε", "draft gap")
phys_set = grab(FACTS, r"physics VLM judge \(accept if [^)]*\), (jar/castle × \d+ seeds)", "physics judge set")
# facts §7 (corrected 10/12): verify step = VAE decode + judge (judge-only time not logged); τ=4 times suspect
phys_cost = grab(FACTS, r"judge-only time not logged\) ([\d–]+) s per clip at τ=5", "physics verify-step cost")
# facts §8: Draft-then-Finish, drift head, ESRGAN, other closed splits
df_sigma = grab(FACTS, r"14B one forward from σ ([\d.–]+)\)", "DtF sigma range")
df_q, df_teacher_q = grab(FACTS, r"quality ([\d.]+) = draft level \(teacher ([\d.]+)\)", "DtF quality")
auroc = grab(FACTS, r"drift head \(AUROC ([\d.]+) on unseen scenes\)", "drift head AUROC")
redraw_time = grab(FACTS, r"redrawn chunks flagged again; no gain, time ×(\d+)", "redraw time factor")
sp_gain = grab(FACTS, r"Sequence parallel 2 GPUs over PCIe: (≤[\d.]+×)", "sequence parallel")
offload = grab(FACTS, r"Commit offload to a second GPU: hallucinated scenes \((\d+/\d+) clips\)", "commit offload")
assert "fused commit matched separate commit there" in FACTS, "facts.md §8 DtF wording changed"
assert "Real-ESRGAN worse than bicubic" in FACTS, "facts.md §8 ESRGAN wording changed"
assert "Adaptive 576→832 switch: degenerates into scene routing" in FACTS, "facts.md §8 switch wording changed"
# facts §10: still pending?
cw_b43_pending = "Chunk-wise adaptive + fused/partial commit on bench-43" in grab(
    FACTS, r"## 10\. Pending(.*)$", "pending section")
# README §T: Draft-then-Finish prompt set and fused-vs-separate quality
df_prompts, df_seed = grab(README, r"(\d+) prompts, seed (\d+), full-VAE video", "DtF prompt set")
df_fq, df_sq = grab(README, r"fused and separate commits give the same quality \(([\d.]+) vs ([\d.]+)\)", "DtF fused vs sep")
# README §L4: verify + redraw table
l4 = grab(README, r"### L4\. Verify \+ redraw(.*?)\n## ", "README L4")
l4_base_s, l4_base_flag = grab(l4, r"\| 14B 2-step, scored only \| ([\d.]+) \| ([\d.]+) \|", "L4 base row")
l4_red_s, l4_red_flag = grab(l4, r"\| 2-step \+ verify → dense 4-step redraw \| ([\d.]+) \| ([\d.]+) \|", "L4 redraw row")
l4_scenes = grab(l4, r"\((?:[^;]*); (\d+) scenes ×", "L4 scenes")

# bench-43 (43 prompts x 2 seeds = 86 clips / arm; fps under load)
T, S = B43["teacher"], B43["student"]
SD07, SD22 = B43["sdvg07"], B43["sdvg22"]
O832, O704 = B43["ours832"], B43["ours704"]
CWS = B43["cw07_safe"]  # ours: teacher-parity mode (chunk-wise adaptive 0.7 + safe commit), facts §12
for a in (T, S, SD07, SD22, O832, O704, CWS):
    assert a["clips"] == 86, a
for a in (O832, O704, CWS):
    assert a["paired_n"] == 86, a

# ---------------------------------------------------------------- style
C_TEACHER, C_STUDENT = "#1f1f1f", "#e07b00"
C_OURS832, C_OURS704 = "#7b4bd1", "#d81b60"
C_WORKS, C_CLOSED = "#1e8e3e", "#8a8a8a"
INK, INK2, MUTED, GRID = "#1f1f1f", "#444444", "#8a8a8a", "#e6e6e6"
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})

# column geometry (inches)
W = 13.33
X_CHIP, W_CHIP = 0.30, 0.98
X_STRAT = 1.42
X_DEV, W_DEV = 4.10, 1.40
X_CLD, W_CLD = 5.90, 1.56
X_EVID = 7.66
X_RIGHT = 13.05
IND = 0.16  # indent of evidence continuation lines

# ---------------------------------------------------------------- rows
# lines: list of (text, indent?) ; head: str or (colour, str)
rows = [
    dict(status="closed", name=["Speculative decoding"],
         sub=["device drafts, cloud verifies:", "SDVG + actions; ImageReward or", "physics-VLM verifier"],
         dev=("1.3B drafts\neach chunk", "gen"), cld="ImageReward or\nphysics VLM verifies;\n14B regenerates\nrejects + commits",
         bh=0.76, arrows="both",
         head=f"Draft is {gap_sigma} σ from the 14B; only {gap_eps} % of tokens within ε",
         lines=[(f"SDVG τ −0.7: {SD07['fps']:.1f} fps, slower than the teacher ({T['fps']:.1f} fps)", 0),
                (f"SDVG τ −2.2: {SD22['fps']:.1f} fps, drift {SD22['drift']:.1f} (student {S['drift']:.1f}, "
                 f"teacher {T['drift']:.1f})", 0),
                (f"physics-VLM verify step (decode + judge): {phys_cost} s per clip at τ=5", 0)]),
    dict(status="closed", name=["Draft-then-Finish"],
         sub=["device drafts, cloud corrects:", f"14B one forward from σ {df_sigma}"],
         dev=("1.3B drafts\nat 832", "gen"), cld="14B re-noises +\nfinishes (1 fwd)", bh=0.56, arrows="both",
         head=f"Quality {df_q} = draft level (teacher {df_teacher_q})",
         lines=[("the finish keeps the draft's layout and camera path", 0),
                (f"side result, in DtF: fused = separate commit ({df_fq} vs {df_sq}) → reused in ours,", 0),
                (f"+ partial commit: fused alone has boundary jumps (704: {j704_fused} % vs separate {j704_sep} %)", 1)]),
    dict(status="closed", name=["Verify + redraw"],
         sub=["learned drift head on 14B features,", f"AUROC {auroc} on unseen scenes"],
         dev=("VAE decode", "dec"), cld="14B draws chunk;\nhead flags → redraw", bh=0.56, arrows="down",
         head=f"Redrawn chunks are flagged again: no gain, time ×{redraw_time}",
         lines=[(f"chunks flagged (of 12): {l4_base_flag} before, {l4_red_flag} after redraw", 0),
                (f"{l4_base_s} → {l4_red_s} s per clip ({l4_scenes} scenes × 2 seeds, 14B 832)", 0)]),
    dict(status="works", name=["Target acceleration"],
         sub=["speeds up the cloud 14B: 2 steps +", "fused/partial commit + systems stack", "(chunk-wise adaptive: separate config)"],
         dev=("VAE decode", "dec"), cld="14B at 832,\n2 steps", bh=0.56, arrows="down",
         head=(C_OURS832, f"ours 832: {O832['fps']:.1f} fps vs teacher {T['fps']:.1f} fps"),
         lines=[(f"Δ quality {sgn(O832['d_quality'])} ± {O832['d_quality_se']:.2f}; drift {O832['drift']:.1f} vs "
                 f"teacher {T['drift']:.1f} (Δ {sgn(O832['d_drift'], 1)} ± {O832['d_drift_se']:.1f})", 0),
                (f"systems stack + 2 steps: {s0} → {s7} s per clip (systems only, 4 steps: → {s3} s)", 0),
                (f"chunk-wise adaptive + safe commit: {cwq_fps} vs teacher {tq_fps} fps quiet = {cwq_x};", 0),
                (f"bench-43: Δ quality {sgn(CWS['d_quality'])} ± {CWS['d_quality_se']:.2f}, Δ drift "
                 f"{sgn(CWS['d_drift'], 1)} ± {CWS['d_drift_se']:.1f} (paired)", 1)]),
    dict(status="works", name=["Low-res cloud (704) +", "device bicubic upsample"],
         sub=["cloud: 14B at 704, 2 steps;", "device: VAE decode + bicubic, no SR"],
         dev=("VAE decode +\nbicubic (no 1.3B)", "dec"), cld="14B at 704,\n2 steps", bh=0.56, arrows="down",
         head=(C_OURS704, f"ours 704 (realtime): {O704['fps']:.1f} fps vs teacher {T['fps']:.1f} fps"),
         lines=[(f"Δ quality {sgn(O704['d_quality'])} ± {O704['d_quality_se']:.2f}; drift {O704['drift']:.1f} vs "
                 f"teacher {T['drift']:.1f} (Δ {sgn(O704['d_drift'], 1)} ± {O704['d_drift_se']:.1f})", 0),
                (f"cloud {cloud704} s per chunk (under load)", 0),
                (f"device SR is closed: 1.3B refiner Δ quality {neg(ref_d)} ± {ref_se}; Real-ESRGAN < bicubic", 0)]),
]

LS_EV = 0.205     # evidence line spacing
LS_NAME, LS_SUB, GAP_NS = 0.215, 0.19, 0.235
for r in rows:
    ev_span = LS_EV * len(r["lines"])
    st_span = LS_NAME * (len(r["name"]) - 1) + GAP_NS + LS_SUB * (len(r["sub"]) - 1)
    r["ev_span"], r["st_span"] = ev_span, st_span
    r["h"] = max(ev_span, st_span, r["bh"]) + 0.32

# ---------------------------------------------------------------- vertical layout (distance from top, inches)
D_TITLE, D_SUB1, D_SUB2 = 0.32, 0.64, 0.86
D_HEAD = 1.22
D_ROWS = D_HEAD + 0.22
d = D_ROWS
for r in rows:
    r["d_top"] = d
    d += r["h"]
D_STRIP = d + 0.30
D_FOOT = D_STRIP + 0.44
LS_FOOT = 0.185
foot = [
    f"Bench-43 (default set): 38 StEvo camera-WM tasks + 5 easy scenes × 2 seeds = {T['clips']} clips/arm, Qwen3-VL-32B judge, "
    f"Δ = paired vs teacher ± SE, fps under load ({n_load} concurrent jobs).",
    f"Chunk-wise speed: quiet steady s/chunk, 10/12 timing pass, jar + castle (teacher {tq_s} s/chunk = {tq_fps} fps quiet, "
    f"3.9 under load;",
    f"the older “4.9 fps quiet” was a whole-clip average incl. cheap early chunks). Refiner Δ: 12 prompts × 3 seeds, "
    f"{ref_n} pairs, same judge.",
    f"DtF: {df_prompts} prompts, seed {df_seed}. Physics judge: {phys_set}. Verify + redraw: {l4_scenes} scenes × 2 seeds. "
    f"Systems stack: 14B, {n_sys} prompts. Draft gap: per latent token.",
    f"Cloud/edge timing: ours fps = cloud throughput (12 frames / cloud s per chunk), device decode assumed pipelined on a "
    f"separate GPU (one shared GPU: 704 cloud + VAE = {shared_fps} fps);",
    f"device VAE ({vae704} ms/chunk at 704) timed on the server GPU, not on an edge device; network transfer excluded. "
    f"All timings: one RTX PRO 6000 Blackwell.",
    "Sources: paper/1012/bench43_summary.json; paper/meeting_1012/facts.md §0, §1, §3, §5–§8, §12, §15; "
    "paper/1012/README.md §L4, §R, §T.",
]
H = D_FOOT + LS_FOOT * (len(foot) - 1) + 0.20

fig = plt.figure(figsize=(W, H), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def Y(dist):
    return H - dist


texts = []  # (artist, right bound, (y_low, y_high) band or None)


def T_(x, y, s, right=X_RIGHT, band=None, **kw):
    t = ax.text(x, y, s, **kw)
    texts.append((t, right, band))
    return t


# ---------------------------------------------------------------- title block
T_(0.30, Y(D_TITLE), f"Only splits that keep the 14B generating survive: ours 832 at {O832['fps']:.1f} fps "
   f"and ours 704 at {O704['fps']:.1f} fps", fontsize=13, fontweight="bold", color=INK, va="center")
T_(0.30, Y(D_SUB1), "Cloud/edge splits and acceleration routes explored to date, including the splits proposed at the "
   "10/10 meeting. The 1.3B is the only device model tested.", fontsize=10, color=INK2, va="center")
T_(0.30, Y(D_SUB2), "Box edge: orange = 1.3B generates content · grey = device only decodes · black = 14B.   "
   "Arrows = direction of data flow (→ cloud: draft chunk · ← device: finished chunk latents).",
   fontsize=9.5, color=INK2, va="center")

# header
y_head = Y(D_HEAD)
for x, s, ha in [(X_CHIP + W_CHIP / 2, "Status", "center"), (X_STRAT, "Split / route", "left"),
                 (X_DEV + W_DEV / 2, "Device (edge)", "center"), (X_CLD + W_CLD / 2, "Cloud", "center"),
                 (X_EVID, "Evidence  (fps ↑ · VLM quality 1–10 ↑ · VLM drift 0–100 ↓)", "left")]:
    T_(x, y_head, s, fontsize=10, fontweight="bold", color=INK2, ha=ha, va="center")
ax.plot([0.30, X_RIGHT], [Y(D_ROWS - 0.05)] * 2, color="#bdbdbd", lw=1.0)

# ---------------------------------------------------------------- rows
for i, r in enumerate(rows):
    yt = Y(r["d_top"])
    yb = yt - r["h"]
    yc = (yt + yb) / 2
    band = (yb, yt)
    works = r["status"] == "works"
    if works:
        ax.add_patch(Rectangle((0.22, yb + 0.03), X_RIGHT + 0.06 - 0.22, r["h"] - 0.06,
                               facecolor="#eef7f0", edgecolor="none", zorder=0))
    ax.plot([0.30, X_RIGHT], [yb, yb], color=GRID, lw=1.0, zorder=1)

    # status chip
    ax.add_patch(FancyBboxPatch((X_CHIP, yc - 0.16), W_CHIP, 0.32, boxstyle="round,pad=0,rounding_size=0.10",
                                facecolor=C_WORKS if works else C_CLOSED, edgecolor="none", zorder=2))
    T_(X_CHIP + W_CHIP / 2, yc, ("✓ WORKS" if works else "✕ CLOSED"), fontsize=9.5, fontweight="bold",
       color="white", ha="center", va="center", zorder=3, right=X_STRAT, band=band)

    # strategy name + sub-lines (block centred on the row)
    y0 = yc + r["st_span"] / 2
    for k, s in enumerate(r["name"]):
        T_(X_STRAT, y0 - LS_NAME * k, s, fontsize=11, fontweight="bold", color=INK, va="center",
           right=X_DEV - 0.05, band=band)
    y1 = y0 - LS_NAME * (len(r["name"]) - 1) - GAP_NS
    for k, s in enumerate(r["sub"]):
        T_(X_STRAT, y1 - LS_SUB * k, s, fontsize=9, color=INK2, va="center", right=X_DEV - 0.05, band=band)

    # device box / cloud box
    dtxt, dkind = r["dev"]
    bh = r["bh"]
    ax.add_patch(FancyBboxPatch((X_DEV, yc - 0.28), W_DEV, 0.56, boxstyle="round,pad=0,rounding_size=0.07",
                                facecolor="#fdf1e3" if dkind == "gen" else "#f4f4f4",
                                edgecolor=C_STUDENT if dkind == "gen" else "#9e9e9e",
                                lw=2.0 if dkind == "gen" else 1.2, zorder=2))
    T_(X_DEV + W_DEV / 2, yc, dtxt, fontsize=9, color=INK, ha="center", va="center", linespacing=1.15,
       zorder=3, right=X_DEV + W_DEV, band=(yc - 0.28, yc + 0.28))
    ax.add_patch(FancyBboxPatch((X_CLD, yc - bh / 2), W_CLD, bh, boxstyle="round,pad=0,rounding_size=0.07",
                                facecolor="#ededed", edgecolor=C_TEACHER, lw=1.6, zorder=2))
    T_(X_CLD + W_CLD / 2, yc, r["cld"], fontsize=9, color=INK, ha="center", va="center", linespacing=1.15,
       zorder=3, right=X_CLD + W_CLD, band=(yc - bh / 2, yc + bh / 2))

    # arrows: device -> cloud = draft chunk; cloud -> device = finished chunk latents
    xa0, xa1 = X_DEV + W_DEV + 0.04, X_CLD - 0.04
    kw = dict(arrowstyle="-|>", mutation_scale=11, lw=1.4, color="#555555", zorder=3)
    if r["arrows"] == "both":
        ax.add_patch(FancyArrowPatch((xa0, yc + 0.09), (xa1, yc + 0.09), **kw))
        ax.add_patch(FancyArrowPatch((xa1, yc - 0.09), (xa0, yc - 0.09), **kw))
    else:
        ax.add_patch(FancyArrowPatch((xa1, yc), (xa0, yc), **kw))

    # evidence (block centred on the row)
    head = r["head"]
    y_head_ev = yc + r["ev_span"] / 2
    if isinstance(head, tuple):
        col, s = head
        ax.add_patch(Rectangle((X_EVID, y_head_ev - 0.065), 0.13, 0.13, facecolor=col, edgecolor="none", zorder=3))
        T_(X_EVID + 0.2, y_head_ev, s, fontsize=10.5, fontweight="bold", color=INK, va="center", band=band)
    else:
        T_(X_EVID, y_head_ev, head, fontsize=10.5, fontweight="bold", color=INK, va="center", band=band)
    for k, (s, indent) in enumerate(r["lines"]):
        T_(X_EVID + IND * indent, y_head_ev - LS_EV * (k + 1), s, fontsize=9, color=INK2, va="center", band=band)

# ---------------------------------------------------------------- also-closed strip
y_strip = Y(D_STRIP)
T_(0.30, y_strip, "Also closed:", fontsize=9.5, fontweight="bold", color=INK2, va="center")
items = ["adaptive 576→832 switch: degenerates into scene routing",
         f"commit offload to a 2nd GPU: hallucinated scenes ({offload} clips)",
         f"sequence parallel, 2 GPUs over PCIe: {sp_gain}"]
fig.canvas.draw()
rend = fig.canvas.get_renderer()
inv = ax.transData.inverted()
x = 1.40
for s in items:
    t = T_(x + 0.10, y_strip, s, fontsize=9, color=INK2, va="center", zorder=3)
    bb = t.get_window_extent(rend)
    (x0, _), (x1, _) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    ax.add_patch(FancyBboxPatch((x0 - 0.08, y_strip - 0.13), (x1 - x0) + 0.16, 0.26,
                                boxstyle="round,pad=0,rounding_size=0.08", facecolor="#f2f2f2",
                                edgecolor="#c8c8c8", lw=0.8, zorder=2))
    x = x1 + 0.18

# ---------------------------------------------------------------- footer
for k, s in enumerate(foot):
    T_(0.30, Y(D_FOOT + LS_FOOT * k), s, fontsize=9, color=MUTED, va="center")

# ---------------------------------------------------------------- layout checks
fig.canvas.draw()
rend = fig.canvas.get_renderer()
bad = []
boxes = []
for t, right, band in texts:
    bb = t.get_window_extent(rend)
    (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    if x1 > right + 1e-3 or x0 < 0.05 or y0 < 0.02 or y1 > H - 0.02:
        bad.append(f"overflow ({x1:.2f} > {right:.2f}): {t.get_text()[:60]!r}")
    if band is not None and (y0 < band[0] - 1e-3 or y1 > band[1] + 1e-3):
        bad.append(f"outside its row/box ({y0:.2f}-{y1:.2f} vs {band[0]:.2f}-{band[1]:.2f}): {t.get_text()[:50]!r}")
    boxes.append((x0, y0, x1, y1, t.get_text()))
for a in range(len(boxes)):
    for b in range(a + 1, len(boxes)):
        A, B = boxes[a], boxes[b]
        if A[0] < B[2] - 0.01 and B[0] < A[2] - 0.01 and A[1] < B[3] - 0.01 and B[1] < A[3] - 0.01:
            bad.append(f"overlap: {A[4][:40]!r} / {B[4][:40]!r}")
for s in bad:
    print("[layout]", s)

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / f"{FIG_ID}.png", dpi=200, facecolor="white")
fig.savefig(OUT / f"{FIG_ID}.svg", facecolor="white")
print(f"canvas {W:.2f} x {H:.2f} in")
print("wrote", OUT / f"{FIG_ID}.png", OUT / f"{FIG_ID}.svg", "| layout issues:", len(bad))
