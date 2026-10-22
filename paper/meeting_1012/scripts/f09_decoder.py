"""f09_decoder - Decoder choice on the device: full Wan2.1 VAE vs TAEHV (taew2_1), at 704 (ours 704 = fused + partial commit 1).

Left : decode time per chunk at 704 (full VAE vs TAEHV) against the 704 cloud chunk time. The decode is hidden only if the
       device has its own GPU of the same class as the one the decode was timed on (RTX PRO 6000 Blackwell). All fps in this
       figure are COMPUTED from measured times (no end-to-end two-GPU run with a decoder in the loop exists).
Right: paired change TAEHV - full VAE in VLM quality and VLM drift score (36 clips, judge at 448 px and at 832 px), mean +/- SE.
       Both decoders decode the same cloud latents, so the rollout is identical; a higher drift score is the judge reading
       decoder artifacts as drift, not rollout drift.

Data
- Timings: paper/meeting_1012/facts.md section 6 (README W2; scripts/taehv_ablation.py) and section 1 (cloud 0.761 s/chunk, under load).
  No raw timing log of scripts/taehv_ablation.py exists, so the numbers are parsed from facts.md. The per-chunk divisor is
  parsed from scripts/taehv_ablation.py itself (it decodes whole clips in one call and divides by the chunk count).
- Quality / drift: recomputed per clip from the raw VLM scores
    paper/1012/vlm_scores_diag.json     labels pc1_704 (full VAE) / pc1_704_taehv   (judge 448 px)
    paper/1012/vlm_scores_diag832.json  labels hi_pc1_704_vae      / hi_pc1_704_taehv (judge 832 px)
  paired by clip directory, SE = std(ddof=1)/sqrt(n).
- Frames per clip: counted with cv2 on the scored videos (falls back to omitting the count if the videos are gone).
Run: CUDA_VISIBLE_DEVICES="" /home/hk28965/specwm/venv/bin/python paper/meeting_1012/scripts/f09_decoder.py
"""
import json, os, re
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/hk28965/specwm"
OUT = ROOT + "/paper/meeting_1012/figs"
FACTS = ROOT + "/paper/meeting_1012/facts.md"
ABL = ROOT + "/scripts/taehv_ablation.py"
ID = "f09_decoder"

C_VAE = "#d81b60"      # ours 704 (the canonical ours-704 arm is decoded with the full VAE)
C_TAE = "#8a8a8a"      # anything else: grey
C_REF = "#424242"
C_DELTA = "#5f5f5f"
GRID = "#e6e6e6"
INK2 = "#4a4a4a"
MUTED = "#616161"      # footer / notes: dark enough for a projector

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11.5, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.edgecolor": "#9e9e9e", "axes.linewidth": 0.8,
    "svg.fonttype": "none",
})

# ---------------------------------------------------------------- timings and context from facts.md
facts = open(FACTS).read()
sec = lambda a, b: facts.split(f"## {a}.")[1].split(f"## {b}.")[0]
sec0, sec1, sec2, sec6 = sec(0, 1), sec(1, 2), sec(2, 3), sec(6, 7)


def need(pat, text, what):
    m = re.search(pat, text)
    assert m, f"{what} not found in facts.md"
    return m.groups()


vae_ms, tae_ms = need(r"At 704: full Wan2\.1 VAE (\d+(?:\.\d+)?) ms/chunk vs TAEHV \(taew2_1\) (\d+(?:\.\d+)?) ms/chunk", sec6, "704 decoder timings")
(cloud_ms,) = need(r"faster than the cloud chunk \((\d+) ms\)", sec6, "cloud chunk time (section 6)")
sh_vae_s, sh_vae_fps, sh_tae_s, sh_tae_fps = need(
    r"cloud \+ VAE ([\d.]+) s → ([\d.]+) fps; cloud \+ TAEHV ([\d.]+) s → ([\d.]+) fps", sec6, "shared-GPU numbers")
cloud_s, pipe_fps = need(r"Cloud ([\d.]+) s/chunk → ([\d.]+) fps \(measured under load", sec1, "ours 704 cloud fps")
vae832_ms, tae832_ms = need(r"At 832 the full VAE costs (\d+) ms/chunk \(TAEHV (\d+) ms\)", sec6, "832 decoder timings")
(cloud832_s,) = need(r"same at 832 \(max_area \d+\): ([\d.]+) s/chunk", sec1, "ours 832 cloud s/chunk")
(chunk_frames,) = need(r"Chunk = 3 latent frames = (\d+) video frames", sec0, "frames per chunk")
(auroc,) = need(r"clip-level AUROC ([\d.]+) vs human", sec2, "drift-prompt AUROC")
(gpu_gb,) = need(r"RTX PRO 6000 Blackwell Server Edition \((\d+) GB\)", facts, "timing GPU")

# the "computed" numbers really are what we say they are
assert abs(float(cloud_s) * 1000 - float(cloud_ms)) < 0.5, (cloud_s, cloud_ms)
assert abs(int(chunk_frames) / float(cloud_s) - float(pipe_fps)) < 0.05, "pipelined fps is not 12 / cloud s"
assert abs((float(cloud_ms) + float(vae_ms)) / 1000 - float(sh_vae_s)) < 0.005, "shared VAE s is not cloud + VAE"
assert abs((float(cloud_ms) + float(tae_ms)) / 1000 - float(sh_tae_s)) < 0.0005, "shared TAEHV s is not cloud + TAEHV"

# per-chunk divisor used by the timing script (whole-clip decode / n_chunks)
abl = open(ABL).read()
divs = set(re.findall(r"\(time\.time\(\) - t0\) / (\d+)\)", abl))
assert len(divs) == 1, divs
n_div = divs.pop()

# ---------------------------------------------------------------- paired VLM deltas from raw scores
def load(path, label):
    d = json.load(open(os.path.join(ROOT, path)))
    out = {}
    for k, v in d.items():
        if v["label"] == label:
            clip = k.split("/cloud/")[0] if "/cloud/" in k else os.path.dirname(k)
            out[clip] = (k, v)
    return out


judges = [("judge 448 px", "paper/1012/vlm_scores_diag.json", "pc1_704", "pc1_704_taehv"),
          ("judge 832 px", "paper/1012/vlm_scores_diag832.json", "hi_pc1_704_vae", "hi_pc1_704_taehv")]
res, videos = {}, set()
for name, path, la, lb in judges:
    A, B = load(path, la), load(path, lb)
    keys = sorted(set(A) & set(B))
    assert len(keys) == len(A) == len(B), (name, len(A), len(B), len(keys))
    videos |= {A[k][0] for k in keys} | {B[k][0] for k in keys}
    r = {"n": len(keys)}
    for met in ("quality", "drift"):
        a = np.array([A[k][1][met] for k in keys], float)
        b = np.array([B[k][1][met] for k in keys], float)
        dd = b - a
        r[met] = dict(vae=a.mean(), tae=b.mean(), d=dd.mean(), se=dd.std(ddof=1) / np.sqrt(len(dd)),
                      worse=int((dd < 0).sum() if met == "quality" else (dd > 0).sum()),
                      same=int((dd == 0).sum()), better=int((dd > 0).sum() if met == "quality" else (dd < 0).sum()))
    res[name] = r
    print(name, json.dumps({k: (v if k == "n" else {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
                            for k, v in r.items()}))
N = res["judge 448 px"]["n"]

# frames per scored clip (counted, not assumed: the requested frame_num 161 decodes to fewer frames)
frames_txt = f"{n_div} chunks"
try:
    import cv2
    cnt = Counter()
    for v in sorted(videos):
        p = os.path.join(ROOT, v)
        if os.path.exists(p):
            cap = cv2.VideoCapture(p); cnt[int(cap.get(cv2.CAP_PROP_FRAME_COUNT))] += 1; cap.release()
    print("frame counts of scored videos:", dict(cnt), "of", len(videos))
    if len(cnt) == 1:
        frames_txt = f"{next(iter(cnt))} frames ({n_div} chunks)"
except ImportError:
    pass


def um(s):
    """typographic minus for display"""
    return s.replace("-", "−")


# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(12, 6.75))
gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 0.95, 0.95], left=0.085, right=0.985, top=0.69, bottom=0.27, wspace=0.42)
axL = fig.add_subplot(gs[0]); axQ = fig.add_subplot(gs[1]); axD = fig.add_subplot(gs[2])


def style(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(colors="#333333")


def panel_title(ax, main, sub=None):
    """main line(s) in 11.5 pt, optional grey sub line(s) in 9.5 pt, stacked above the axes (offsets in points)"""
    y0 = 7
    if sub:
        ax.annotate(sub, xy=(0, 1), xycoords="axes fraction", xytext=(0, y0), textcoords="offset points",
                    ha="left", va="bottom", fontsize=9.5, color=INK2, linespacing=1.3)
        y0 += 9.5 * 1.3 * (sub.count("\n") + 1) + 3
    ax.annotate(main, xy=(0, 1), xycoords="axes fraction", xytext=(0, y0), textcoords="offset points",
                ha="left", va="bottom", fontsize=11.5, color="#111111", linespacing=1.25)


# --- left: decode time
style(axL)
labels = ["Full Wan2.1 VAE\n(bf16)", "TAEHV taew2_1\n(fp16)"]
vals = [float(vae_ms), float(tae_ms)]
xs = np.arange(2)
axL.bar(xs, vals, width=0.52, color=[C_VAE, C_TAE], zorder=3)
axL.text(0, vals[0] - 18, f"{vae_ms} ms", ha="center", va="top", fontsize=10.5, color="white", fontweight="bold", zorder=5)
axL.text(1, vals[1] + 18, f"{tae_ms} ms", ha="center", va="bottom", fontsize=10.5, color="#222222", fontweight="bold")
axL.axhline(float(cloud_ms), color=C_REF, ls=(0, (5, 3)), lw=1.4, zorder=4)
axL.text(1.62, float(cloud_ms) + 14, f"cloud 14B chunk at 704 = {cloud_ms} ms\n(fused + partial commit, measured under load)",
         ha="right", va="bottom", fontsize=9, color=C_REF)
# gap between the top of the VAE bar and the cloud chunk line
axL.annotate("", xy=(0, float(cloud_ms)), xytext=(0, float(vae_ms)),
             arrowprops=dict(arrowstyle="<->", color=INK2, lw=1, shrinkA=0, shrinkB=0), zorder=4)
axL.text(0.07, (float(cloud_ms) + float(vae_ms)) / 2,
         f"Separate device GPU of this class:\nthe VAE decode ends before the next\ncloud chunk, so pipelined fps stays\n"
         f"cloud-bound: {chunk_frames} / {cloud_s} s = {pipe_fps} fps\n(computed)",
         ha="left", va="center", fontsize=9, color=INK2, linespacing=1.3)
axL.text(1.6, 375, f"One shared GPU (serial), computed\nas the sum of the measured times:\n"
                   f"VAE  {sh_vae_s} s → {sh_vae_fps} fps\n"
                   f"TAEHV  {sh_tae_s} s → {sh_tae_fps} fps",
         ha="right", va="top", fontsize=9, color=INK2, linespacing=1.4,
         bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="#d6d6d6", lw=0.8))
axL.set_xticks(xs, labels)
axL.set_xlim(-0.5, 1.65)
axL.set_ylim(0, 1000)
axL.set_ylabel(f"Decode time per chunk at 704 (ms)\n(whole-clip decode ÷ {n_div}; ↓ lower is better)")
panel_title(axL, "Speed: on a device GPU of this class,\nboth decoders hide under the cloud chunk")


# --- right: paired deltas (one panel per metric: different scales, no dual axis)
def delta_panel(ax, met, ylabel, fmt):
    style(ax)
    names = list(res)
    xs = np.arange(len(names))
    vals = [res[n][met]["d"] for n in names]; ses = [res[n][met]["se"] for n in names]
    ax.axhline(0, color="#333333", lw=1, zorder=2)
    ax.errorbar(xs, vals, yerr=ses, fmt="o", ms=9, color=C_DELTA, ecolor=C_DELTA, elinewidth=1.6, capsize=6, zorder=4,
                markeredgecolor="white", markeredgewidth=1.5)
    for i, (x, v, s) in enumerate(zip(xs, vals, ses)):
        side = 1 if i == 0 else -1          # first label right of its point, second left: keeps clear of the other error bar
        ax.text(x + 0.13 * side, v, um(f"{v:+{fmt}} ± {s:{fmt}}"), ha="left" if side > 0 else "right", va="center",
                fontsize=10, color="#222222", fontweight="bold")
    ticks = [f"{n}\nfull VAE {res[n][met]['vae']:{fmt}}\nTAEHV {res[n][met]['tae']:{fmt}}" for n in names]
    ax.set_xticks(xs, ticks)
    for t in ax.get_xticklabels():
        t.set_linespacing(1.35)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel(ylabel)


delta_panel(axQ, "quality", "Δ VLM quality, TAEHV − full VAE\n(1–10 scale; ↑ higher is better)", ".2f")
axQ.set_ylim(-1.5, 0.3)
axQ.text(1.5, 0.05, "0 = no change", ha="right", va="bottom", fontsize=9, color=MUTED)
panel_title(axQ, "Quality score:\njudge rates TAEHV lower", "prompt not validated\nagainst humans")
delta_panel(axD, "drift", "Δ VLM drift score, TAEHV − full VAE\n(0–100 scale; ↓ lower is better)", ".1f")
axD.set_ylim(-3, 17)
axD.text(1.5, -0.4, "0 = no change", ha="right", va="top", fontsize=9, color=MUTED)
panel_title(axD, "Drift score:\njudge rates TAEHV higher",
            "same latents: decoder artifacts\nread as drift, not rollout drift")

# --- title, notes, footer
q448, q832 = res["judge 448 px"]["quality"], res["judge 832 px"]["quality"]
d448, d832 = res["judge 448 px"]["drift"], res["judge 832 px"]["drift"]
fig.text(0.012, 0.975,
         um(f"If the device GPU matches an RTX PRO 6000, keep the full VAE: {vae_ms} ms hides under the {cloud_ms} ms cloud chunk;\n"
            f"TAEHV ({tae_ms} ms) costs {q448['d']:.2f} / {q832['d']:.2f} VLM quality (1–10 scale, judge 448 / 832 px)"),
         ha="left", va="top", fontsize=13, fontweight="bold", color="#111111", linespacing=1.3)
fig.text(0.012, 0.888,
         f"Ours 704 (fused + partial commit); the same cloud latents decoded twice on the device. Δ = TAEHV − full VAE, paired by clip, n = {N}, mean ± SE.\n"
         f"Clips where TAEHV is worse / better (rest tied): quality {q448['worse']} / {q448['better']} at judge 448 and "
         f"{q832['worse']} / {q832['better']} at judge 832; drift score {d448['worse']} / {d448['better']} at judge 448 and "
         f"{d832['worse']} / {d832['better']} at judge 832.",
         ha="left", va="top", fontsize=9.5, color=INK2, linespacing=1.5)
fig.text(0.012, 0.018,
         f"Source: 12 prompts × 3 seeds = {N} clips per decoder, {frames_txt} each, ours 704 fused + partial commit 1. VLM judge Qwen3-VL-32B-Instruct, 8 frames per clip\n"
         f"resized to 448 or 832 px wide (paper/1012/vlm_scores_diag.json, vlm_scores_diag832.json). Drift prompt validated against human severe-drift labels (AUROC {auroc});\n"
         f"quality prompt not validated against humans. Decode ms: scripts/taehv_ablation.py on one RTX PRO 6000 Blackwell ({gpu_gb} GB), timed offline on whole clips\n"
         f"(one call per clip ÷ {n_div}; streamed per-chunk decode not timed); load condition not recorded. Cloud {cloud_ms} ms/chunk measured under load (7 concurrent GPU jobs). Chunk = {chunk_frames} frames.\n"
         f"Every fps here is computed from measured times (no end-to-end run with a decoder in the loop). At 832: full VAE {vae832_ms} ms vs TAEHV {tae832_ms} ms per chunk (cloud {cloud832_s} s/chunk).",
         ha="left", va="bottom", fontsize=9, color=MUTED, linespacing=1.4)

os.makedirs(OUT, exist_ok=True)
fig.savefig(f"{OUT}/{ID}.png", dpi=200, facecolor="white")
fig.savefig(f"{OUT}/{ID}.svg", facecolor="white")
print("wrote", f"{OUT}/{ID}.png", f"{OUT}/{ID}.svg")
