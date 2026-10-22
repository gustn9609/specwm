"""f02b_jump_frames: real frames 126..131 around the chunk boundary 128|129 of easy_urban_rain seed 1234,
704 fused (hard cut) vs 704 fused + partial commit 1 (ours 704). The boundary ratio r is recomputed from the
videos with the same definition as scripts/boundary_jump.py.
Re-run:  CUDA_VISIBLE_DEVICES="" /home/hk28965/specwm/venv/bin/python paper/meeting_1012/scripts/f02b_jump_frames.py
"""
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = "/home/hk28965/specwm"
MEET = os.path.join(ROOT, "paper/meeting_1012")
OUT = os.path.join(MEET, "figs", "f02b_jump_frames")
VID = os.path.join(ROOT, "lingbot-world-v2/specwm_out/fused/easy_urban_rain/{}/cloud/spec.mp4")
ROWS = [  # run dir, (bold label, sub label), colour
    ("fc704_s1234", ("704 fused", "no partial commit"), "#9e9e9e"),
    ("pc1_704_s1234", ("ours 704", "fused + partial 1"), "#d81b60"),
]
FRAMES = list(range(126, 132))
B = 128   # boundary between frame B and B+1 (0-based frames; boundaries at 8|9, then every 12 frames)
assert (B - 8) % 12 == 0

INK, INK2, MUTED = "#1f1f1f", "#4d4d4d", "#8a8a8a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})


def read_frames(path):
    cap = cv2.VideoCapture(path)
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    assert fr, "cannot read " + path
    return fr


def boundary_ratio(fr, b):
    """Same as scripts/boundary_jump.py: d = consecutive-frame mean abs diff (grey, 208 px wide)."""
    g = [cv2.cvtColor(cv2.resize(f, (208, int(208 * f.shape[0] / f.shape[1]))), cv2.COLOR_BGR2GRAY).astype(np.float32)
         for f in fr]
    d = np.array([np.abs(g[i + 1] - g[i]).mean() for i in range(len(g) - 1)])
    return d[b] / max(1e-6, np.mean([d[b - 2], d[b - 1], d[b + 1], d[b + 2]])), d


def chunk_of(frame):
    """0-based chunk index: chunk 0 = frames 0-8, then 12 frames per chunk."""
    return 0 if frame <= 8 else 1 + (frame - 9) // 12


data = []
for run, lab, col in ROWS:
    fr = read_frames(VID.format(run))
    r, d = boundary_ratio(fr, B)
    data.append((run, lab, col, fr, r))
    print("%s: %d frames, %dx%d, r(%d|%d) = %.2f" % (run, len(fr), fr[0].shape[1], fr[0].shape[0], B, B + 1, r))
H, W = data[0][3][0].shape[:2]

# ---------------------------------------------------------------- layout (figure inches)
FW_IN, FH_IN = 12.0, 4.6
fig = plt.figure(figsize=(FW_IN, FH_IN), facecolor="white")
left, right_pad = 2.0, 0.15
gap, bgap = 0.05, 0.32          # gap between frames, extra gap at the chunk boundary
tw = (FW_IN - left - right_pad - 5 * gap - (bgap - gap)) / 6
th = tw * H / W
row_gap = 0.4
top = FH_IN - 1.5               # top of the first row of frames
xs = []
x = left
for f in FRAMES:
    xs.append(x)
    x += tw + (bgap if f == B else gap)

r_fused, r_part = data[0][4], data[1][4]
fig.text(0.012, 1 - 0.12 / FH_IN,
         "Fused commit cuts to a new corridor at chunk boundary %d→%d (r = %.2f vs %.2f with partial commit)"
         % (B, B + 1, r_fused, r_part), fontsize=13, fontweight="bold", color=INK, va="top")
fig.text(0.012, 1 - 0.47 / FH_IN,
         "easy_urban_rain, seed 1234, cloud output %d×%d, frames %d–%d (0-based).\n"
         "The two rollouts already differ before frame %d (a different commit gives a different cache state), "
         "so compare continuity across the boundary, not scene content." % (W, H, FRAMES[0], FRAMES[-1], FRAMES[0]),
         fontsize=9.5, color=INK2, va="top", linespacing=1.3)

for ri, (run, lab, col, fr, r) in enumerate(data):
    y0 = top - th - ri * (th + row_gap)
    for f, x in zip(FRAMES, xs):
        ax = fig.add_axes([x / FW_IN, y0 / FH_IN, tw / FW_IN, th / FH_IN])
        ax.imshow(cv2.cvtColor(fr[f], cv2.COLOR_BGR2RGB), interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#d0d0d0")
            s.set_linewidth(0.6)
        if ri == 0:
            ax.set_title("frame %d" % f, fontsize=10, color=INK, pad=3)
    # row label with a colour swatch
    fig.patches.append(Rectangle((0.12 / FW_IN, (y0 + th / 2 + 0.02) / FH_IN), 0.16 / FW_IN, 0.36 / FH_IN,
                                 transform=fig.transFigure, fc=col, ec="none"))
    fig.text(0.36 / FW_IN, (y0 + th / 2 + 0.3) / FH_IN, lab[0], fontsize=10.5, fontweight="bold", color=INK,
             va="center", ha="left")
    fig.text(0.36 / FW_IN, (y0 + th / 2 + 0.08) / FH_IN, lab[1], fontsize=10, color=INK, va="center", ha="left")
    fig.text(0.36 / FW_IN, (y0 + th / 2 - 0.24) / FH_IN, "r = %.2f at %d|%d" % (r, B, B + 1), fontsize=10,
             color=INK2, va="center", ha="left")

# chunk boundary marker between frame B and B+1
xb = (xs[FRAMES.index(B)] + tw + xs[FRAMES.index(B + 1)]) / 2
ytop = top + 0.36
ybot = top - 2 * th - row_gap - 0.05
fig.lines.append(Line2D([xb / FW_IN] * 2, [ybot / FH_IN, ytop / FH_IN], transform=fig.transFigure, color=INK, lw=1.6,
                        ls=(0, (4, 3))))
fig.text(xb / FW_IN, (ybot - 0.05) / FH_IN, "chunk boundary %d | %d" % (B, B + 1), fontsize=10, color=INK,
         ha="center", va="top", fontweight="bold")
fig.text((xb - 0.12) / FW_IN, ytop / FH_IN, "← chunk %d" % chunk_of(B), fontsize=9.5, color=INK2, ha="right",
         va="top")
fig.text((xb + 0.12) / FW_IN, ytop / FH_IN, "chunk %d →" % chunk_of(B + 1), fontsize=9.5, color=INK2, ha="left",
         va="top")

fig.text(0.012, 0.015,
         "Source: lingbot-world-v2/specwm_out/fused/easy_urban_rain/{fc704_s1234, pc1_704_s1234}/cloud/spec.mp4 (16 fps).\n"
         "r = consecutive-frame diff at the boundary ÷ mean of its 4 neighbouring diffs (grey, 208 px; "
         "scripts/boundary_jump.py); jump if r > 1.5.", fontsize=9, color=MUTED, va="bottom", linespacing=1.3)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT + ".png", dpi=200, facecolor="white")
fig.savefig(OUT + ".svg", facecolor="white")
print("wrote", OUT + ".png", OUT + ".svg")
