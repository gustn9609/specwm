"""f08_lowres_curve - Low-resolution isolation: 14B alone (no draft, native output) vs cloud size class.

Points are named by the team's width class (832 / 704 / 576 / 448 / 384 = run labels dg704 ... dg384) and placed on
the x axis by their nominal max_area (pixel budget, log scale). The actual W x H of each clip follows the start
image's aspect (facts.md section 1), e.g. the 448 class is 448x240 for 33/36 clips; the script measures and prints this.

Data (all read from raw files, 12 prompts x 3 seeds = 36 clips per point, judge Qwen3-VL-32B-Instruct):
  832 class : paper/1012/vlm_scores_3seed.json  label '14B 2-step 832'
              videos lingbot-world-v2/specwm_out/cmp10/*/uniform2_s*/video.mp4 (separate run and scoring pass)
              max_area 399360 from facts.md section 1 (this run's loop.json does not record it)
  704 / 576 / 448 / 384 class : paper/1012/vlm_scores_diag.json  labels 'dg<class>_cloud'
              (+ 1.3B refiner: 'dg704_refined', 'dg576_refined')
              videos lingbot-world-v2/specwm_out/fused/*/dg<class>_s*/cloud/spec.mp4, max_area = loop.json 'lo_area'
Paired refiner cost (704 -0.50 +- 0.11, 576 -0.25 +- 0.12) and paired changes vs the 832 class are computed here from
the same per-clip files (matched by prompt/seed).
Re-run: CUDA_VISIBLE_DEVICES="" /home/hk28965/specwm/venv/bin/python paper/meeting_1012/scripts/f08_lowres_curve.py
"""
import collections
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.lines import Line2D

ROOT = "/home/hk28965/specwm"
OUT = f"{ROOT}/paper/meeting_1012/figs/f08_lowres_curve"
F3SEED = f"{ROOT}/paper/1012/vlm_scores_3seed.json"
FDIAG = f"{ROOT}/paper/1012/vlm_scores_diag.json"
FUSED = f"{ROOT}/lingbot-world-v2/specwm_out/fused"
CMP10 = f"{ROOT}/lingbot-world-v2/specwm_out/cmp10"

C_14B = "#1f1f1f"      # teacher 14B colour (the 14B alone)
C_REF = "#e07b00"      # student 1.3B colour (the 1.3B refiner)
C_PEND = "#9e9e9e"
GRID = "#e6e6e6"
FOOT = "#666666"
N_EXPECTED = 36

# (class name, file, label for 14B alone, label for + refiner, run tag on disk, max_area fallback, video glob)
POINTS = [
    ("832", F3SEED, "14B 2-step 832", None, None, 399360, f"{CMP10}/*/uniform2_s*/video.mp4"),
    ("704", FDIAG, "dg704_cloud", "dg704_refined", "dg704", 270336, f"{FUSED}/*/dg704_s*/cloud/spec.mp4"),
    ("576", FDIAG, "dg576_cloud", "dg576_refined", "dg576", 175104, f"{FUSED}/*/dg576_s*/cloud/spec.mp4"),
    ("448", FDIAG, "dg448_cloud", None, "dg448", 114688, f"{FUSED}/*/dg448_s*/cloud/spec.mp4"),
    ("384", FDIAG, "dg384_cloud", None, "dg384", 86016, f"{FUSED}/*/dg384_s*/cloud/spec.mp4"),
]


def load(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def by_label(d, label):
    """{(prompt, seed): {'quality','drift'}} for one label."""
    out = {}
    for k, v in d.items():
        if not isinstance(v, dict) or v.get("label") != label:
            continue
        m = re.search(r"/([^/]+)/[^/]*_s(\d+)/", k)
        key = (m.group(1), m.group(2)) if m else (k, "")
        out[key] = v
    return out


def stats(rows, metric):
    x = np.array([r[metric] for r in rows.values() if r.get(metric) is not None], float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return x.mean(), se, len(x)


def paired(a, b, metric):
    """mean ± SE of b - a over shared (prompt, seed) keys, plus counts worse / equal / better."""
    keys = sorted(set(a) & set(b))
    dif = np.array([b[k][metric] - a[k][metric] for k in keys], float)
    dif = dif[np.isfinite(dif)]
    return dif.mean(), dif.std(ddof=1) / np.sqrt(len(dif)), len(dif), (dif < 0).sum(), (dif == 0).sum(), (dif > 0).sum()


def max_area(tag, fallback):
    """nominal max_area of a run, read from its loop.json files ('lo_area'); fallback = facts.md value."""
    if tag is None:
        return fallback
    vals = set()
    for f in glob.glob(f"{FUSED}/*/{tag}_s*/loop.json"):
        try:
            vals.add(int(json.load(open(f))["lo_area"]))
        except Exception:
            pass
    if len(vals) == 1:
        return vals.pop()
    if vals:
        print(f"WARNING {tag}: several lo_area values {sorted(vals)}; using facts value {fallback}")
    return fallback


def actual_sizes(pattern):
    """Counter of actual (W, H) over the clip videos (header only)."""
    try:
        import cv2
    except Exception:
        return collections.Counter()
    c = collections.Counter()
    for p in sorted(glob.glob(pattern)):
        cap = cv2.VideoCapture(p)
        if cap.isOpened():
            c[(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))] += 1
        cap.release()
    return c


files = {F3SEED: load(F3SEED), FDIAG: load(FDIAG)}
native, refined, pending, sizes = [], [], [], {}
for cls, f, lab, rlab, tag, area_fb, vglob in POINTS:
    x = max_area(tag, area_fb)
    sizes[cls] = actual_sizes(vglob)
    rows = by_label(files[f], lab)
    q, dr = stats(rows, "quality"), stats(rows, "drift")
    if q is None:
        n_vid = len(glob.glob(vglob))
        pending.append((x, cls, n_vid))
        continue
    native.append(dict(x=x, cls=cls, q=q, d=dr, rows=rows))
    if rlab:
        rr = by_label(files[f], rlab)
        refined.append(dict(x=x, cls=cls, q=stats(rr, "quality"), d=stats(rr, "drift"),
                            pq=paired(rows, rr, "quality"), pd=paired(rows, rr, "drift")))

nat = {p["cls"]: p for p in native}
ref = {p["cls"]: p for p in refined}

for p in native:
    print(f"native {p['cls']} class (max_area {p['x']}): quality {p['q'][0]:.2f} ± {p['q'][1]:.2f} (n={p['q'][2]}), "
          f"drift {p['d'][0]:.1f} ± {p['d'][1]:.1f}")
for p in refined:
    print(f"refined {p['cls']}: quality {p['q'][0]:.2f} ± {p['q'][1]:.2f}, drift {p['d'][0]:.1f} ± {p['d'][1]:.1f}; "
          f"paired dq {p['pq'][0]:+.2f} ± {p['pq'][1]:.2f} (n={p['pq'][2]}; worse/equal/better "
          f"{p['pq'][3]}/{p['pq'][4]}/{p['pq'][5]}), dd {p['pd'][0]:+.1f} ± {p['pd'][1]:.1f}")
if "832" in nat:
    for p in native[1:]:
        dq, dd = paired(nat["832"]["rows"], p["rows"], "quality"), paired(nat["832"]["rows"], p["rows"], "drift")
        print(f"paired vs 832 class, {p['cls']} (n={dq[2]}): quality {dq[0]:+.2f} ± {dq[1]:.2f} "
              f"(worse/equal/better {dq[3]}/{dq[4]}/{dq[5]}), drift {dd[0]:+.1f} ± {dd[1]:.1f}")
for cls, c in sizes.items():
    print(f"actual sizes {cls} class: " + ", ".join(f"{w}×{h} ×{n}" for (w, h), n in c.most_common()))
for x, cls, n in pending:
    print(f"PENDING {cls} class: {n} videos on disk, no VLM scores yet")

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "svg.fonttype": "none"})
fig, axes = plt.subplots(1, 2, figsize=(12, 6.75))
fig.subplots_adjust(left=0.065, right=0.985, top=0.765, bottom=0.285, wspace=0.2)

XS = [max_area(p[4], p[5]) for p in POINTS]
XLAB = [f"{p[0]}\n{x / 1000:.0f}k px" for p, x in zip(POINTS, XS)]
XLIM = (min(XS) * 0.86, max(XS) * 1.13)
RX = 1.05  # refiner marker drawn at x * RX (log axis) so its error bar does not sit on the native one


def mfmt(v, f):
    return f.format(v).replace("-", "−")


# label placement per (metric, class): native value labels sit on the side away from the line,
# refiner labels where they clear the neighbouring error bars (checked by eye on the rendered PNG)
NAT_POS = {"q": {"384": "below", "448": "upleft", "576": "upleft", "704": "upleft", "832": "upleft"},
           "d": {"384": "above", "448": "downleft", "576": "downleft", "704": "downleft", "832": "downleft"}}
REF_POS = {"q": {"704": "right", "576": "below"}, "d": {"704": "above", "576": "above"}}
POS = {  # xytext offset (pt), ha, va
    "upleft": ((-6, 4), "right", "bottom"), "downleft": ((-6, -4), "right", "top"),
    "right": ((9, 0), "left", "center"), "left": ((-9, 0), "right", "center"),
}

panels = [
    ("q", "VLM quality (1–10, ↑ higher is better)", "Quality ↑", (4.4, 8.1), "{:.2f}", "{:+.2f} ± {:.2f}"),
    ("d", "VLM drift (0–100, ↓ lower is better)", "Drift ↓", (38, 76), "{:.1f}", "{:+.1f} ± {:.1f}"),
]
OFFICIAL = 480 * 832  # officially supported 480×832 (wan/configs SUPPORTED_SIZES); 720×1280 is off the axis
for ax, (m, ylab, ptitle, ylim, fmt, dfmt) in zip(axes, panels):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color("#9e9e9e")
    ax.set_xscale("log")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*ylim)
    ax.set_xticks(XS)
    ax.set_xticklabels(XLAB)
    for t in ax.get_xticklabels():
        t.set_linespacing(1.35)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("cloud size class (nominal max_area in pixels, log scale)")
    ax.set_ylabel(ylab)
    ax.set_title(ptitle, loc="left", fontsize=12, fontweight="bold", color="#333333")
    span = ylim[1] - ylim[0]

    # officially supported size band
    ax.axvspan(OFFICIAL / 1.07, OFFICIAL * 1.07, color="#ede7f6", zorder=0, lw=0)
    if m == "q":
        ax.text(OFFICIAL, ylim[0] + 0.015 * span, "officially\nsupported\n(480×832)", ha="center", va="bottom",
                fontsize=9, color="#5e35b1")
    else:
        ax.text(OFFICIAL, ylim[1] - 0.015 * span, "officially\nsupported\n(480×832)", ha="center", va="top",
                fontsize=9, color="#5e35b1")

    def put(text, x, y, err, pos, color, weight, size, z):
        if pos == "above":
            ax.annotate(text, (x, y + err), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                        fontsize=size, color=color, fontweight=weight, zorder=z)
        elif pos == "below":
            ax.annotate(text, (x, y - err), xytext=(0, -5), textcoords="offset points", ha="center", va="top",
                        fontsize=size, color=color, fontweight=weight, zorder=z)
        else:
            off, ha, va = POS[pos]
            ax.annotate(text, (x, y), xytext=off, textcoords="offset points", ha=ha, va=va,
                        fontsize=size, color=color, fontweight=weight, zorder=z)

    # 14B alone (native)
    xs = [p["x"] for p in native]
    ys = [p[m][0] for p in native]
    es = [p[m][1] for p in native]
    ax.plot(xs, ys, "-", color=C_14B, lw=2, zorder=3)
    ax.errorbar(xs, ys, yerr=es, fmt="o", ms=8, color=C_14B, mec="white", mew=1.5, ecolor=C_14B,
                elinewidth=1.2, capsize=3, zorder=5)
    for p in native:
        put(mfmt(p[m][0], fmt), p["x"], p[m][0], p[m][1], NAT_POS[m].get(p["cls"], "upleft"),
            "#1f1f1f", "bold", 10, 6)

    # + 1.3B refiner (post-processing on the same clips; closed direction)
    for p in refined:
        xr = p["x"] * RX
        y_nat = nat[p["cls"]][m][0]
        ax.plot([p["x"], xr], [y_nat, p[m][0]], ":", color=C_REF, lw=1.4, zorder=2)
        ax.errorbar([xr], [p[m][0]], yerr=[p[m][1]], fmt="D", ms=7.5, color=C_REF, mec="white", mew=1.2,
                    ecolor=C_REF, elinewidth=1.2, capsize=3, zorder=5)
        pr = p["pq"] if m == "q" else p["pd"]
        put(mfmt(p[m][0], fmt) + "\nΔ " + mfmt(pr[0], dfmt.split(" ± ")[0]) + " ± " + mfmt(pr[1], dfmt.split(" ± ")[1]),
            xr, p[m][0], p[m][1], REF_POS[m].get(p["cls"], "right"), "#8a4b00", "normal", 9, 6)

    # pending classes: hollow markers at the axis bottom
    for x, cls, n in pending:
        yb = ylim[0] + 0.035 * span
        ax.plot([x], [yb], "o", ms=9, mfc="white", mec=C_PEND, mew=1.5, zorder=4, clip_on=False)
        ax.text(x, yb + 0.05 * span, f"pending\n({n}/{N_EXPECTED} clips\nrendered,\nnot scored)",
                ha="center", va="bottom", fontsize=9, color="#757575")

# legend (shared, above the panels)
handles = [
    Line2D([], [], color=C_14B, marker="o", ms=8, mec="white", lw=2,
           label="14B alone, native output (2-step, separate commit, no draft)"),
    Line2D([], [], color=C_REF, marker="D", ms=7, mec="white", lw=0,
           label="+ 1.3B refiner on the same clips (closed direction); Δ = paired change vs native, mean ± SE"),
]
if pending:
    handles.append(Line2D([], [], color=C_PEND, marker="o", ms=8, mfc="white", lw=0, label="pending VLM scoring"))
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 0.848), ncol=len(handles), frameon=False,
           fontsize=9.5, handletextpad=0.4, columnspacing=1.6)

# title = finding (built from the data so it stays correct on re-run)
lo = native[-1]
title = (f"14B alone: quality falls slowly down to the 576 class and steeply below it "
         f"({nat['832']['q'][0]:.2f} → {nat['576']['q'][0]:.2f} → {lo['q'][0]:.2f} at {lo['cls']})")
others = [p["d"][0] for p in native if p["cls"] != lo["cls"]]
sub = (f"Drift jumps only at the {lo['cls']} class ({lo['d'][0]:.1f} vs {min(others):.1f}–{max(others):.1f} for "
       f"{native[0]['cls']}–{native[-2]['cls']}), consistent with Guandao's 10/5 '< 400 px' remark")
if pending:
    sub += "; " + " / ".join(c for _, c, _ in pending) + " pending"
if refined:
    sub += ("\nThe 1.3B refiner lowers quality further at both sizes (paired "
            + ", ".join(f"{mfmt(r['pq'][0], '{:+.2f}')} ± {r['pq'][1]:.2f} at {r['cls']}" for r in refined) + ")")
fig.text(0.012, 0.965, title, fontsize=13, fontweight="bold", ha="left", va="top", color="#1f1f1f")
fig.text(0.012, 0.925, sub, fontsize=11, ha="left", va="top", color="#555555", linespacing=1.35)


def size_note(cls):
    c = sizes.get(cls)
    if not c:
        return None
    (w, h), n = c.most_common(1)[0]
    return f"{cls} class = {w}×{h} for {n}/{sum(c.values())} clips"


eg = size_note("448")
area_list = " / ".join(f"{x:,}" for x in XS)
fig.text(0.012, 0.014,
         "Data: 12 prompts (cmp10 + jar + castle) × 3 seeds = 36 clips per point; mean ± SE over clips; refiner Δ paired "
         "by prompt/seed. No fps on this figure.\n"
         "Judge: Qwen3-VL-32B-Instruct, 8 frames/clip resized to 448 px wide. Officially supported sizes: 480×832 and 720×1280.\n"
         f"Size class = nominal max_area ({area_list} px); actual W×H follows the start-image aspect"
         + (f", e.g. {eg}" if eg else "") + ".\n"
         "832 point: separate run and scoring pass (cmp10/*/uniform2_s*, paper/1012/vlm_scores_3seed.json); "
         "704–384: fused/*/dg<class>_s*/cloud (paper/1012/vlm_scores_diag.json).",
         fontsize=9, color=FOOT, ha="left", va="bottom", linespacing=1.4)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT + ".png", dpi=200, facecolor="white")
fig.savefig(OUT + ".svg", facecolor="white")

# overflow check: every figure-level text must stay inside the canvas
r = fig.canvas.get_renderer()
W = fig.bbox.width
for t in fig.texts + [lg for lg in fig.legends]:
    bb = t.get_window_extent(r)
    if bb.x1 > W - 2 or bb.x0 < 0:
        print(f"WARNING text overflows canvas: x0={bb.x0:.0f} x1={bb.x1:.0f} W={W:.0f}: {type(t).__name__} {str(getattr(t, 'get_text', lambda: '')())[:40]!r}")
print("wrote", OUT + ".png", OUT + ".svg")
