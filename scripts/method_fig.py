"""Method figure (10/15): device drafts each chunk with a few denoising steps at low resolution and shows it; a verifier head on the device's own
features flags chunks; the cloud finishes the flagged chunks with one more step at full resolution and streams the corrected chunk back.
usage: python scripts/method_fig.py OUT.png"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle, Ellipse, Polygon
out = sys.argv[1] if len(sys.argv) > 1 else "method.png"
DEV, CLD, VER, GRY, INK = "#1f8a8a", "#4b5bd8", "#f0882b", "#c9ced6", "#2b2f36"
fig = plt.figure(figsize=(16, 8.2), dpi=150); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 160); ax.set_ylim(0, 82); ax.axis("off"); fig.patch.set_facecolor("white")
def box(x, y, w, h, fc, ec, lw=2, r=3, alpha=1, ls="-"): ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=%g" % r, fc=fc, ec=ec, lw=lw, alpha=alpha, ls=ls))
def arrow(x0, y0, x1, y1, c=INK, lw=2.2, style="-|>", ms=18, cs="arc3,rad=0", ls="-"): ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=c, lw=lw, connectionstyle=cs, ls=ls))
def text(x, y, s, size=13, c=INK, w="normal", ha="center", va="center"): ax.text(x, y, s, fontsize=size, color=c, fontweight=w, ha=ha, va=va)
def grid(x, y, w, h, n, m, fc, ec="white", lw=0.8, alpha=1):   # latent / frame as an n x m grid
    for i in range(n):
        for j in range(m): ax.add_patch(Rectangle((x + j * w / m, y + i * h / n), w / m, h / n, fc=fc, ec=ec, lw=lw, alpha=alpha))
def noisy_grid(x, y, w, h, n, m, seed, base, strength):   # grid with noise texture (how noisy the latent is)
    rng = np.random.default_rng(seed); b = np.array(matplotlib.colors.to_rgb(base))
    for i in range(n):
        for j in range(m):
            c = np.clip(b * (1 - strength) + rng.random(3) * strength, 0, 1); ax.add_patch(Rectangle((x + j * w / m, y + i * h / n), w / m, h / n, fc=c, ec="white", lw=0.5))
def display(x, y, w, h, fc, label, sub, rate, lx=None):
    box(x, y, w, h, "white", INK, lw=2, r=1.5); ax.add_patch(Rectangle((x + 1, y + 1), w - 2, h - 2, fc=fc, ec="none", alpha=0.25)); ax.add_patch(Polygon([[x + w / 2 - 3, y - 2.2], [x + w / 2 + 3, y - 2.2], [x + w / 2 + 1.5, y], [x + w / 2 - 1.5, y]], fc=INK))
    text(lx if lx is not None else x + w / 2, y + h + 2.2, label, 12, w="bold", ha="left" if lx is not None else "center"); text(x + w / 2, y + h / 2 + 1.2, rate, 15, w="bold", c=fc); text(x + w / 2, y + h / 2 - 2.6, sub, 9.5, c="#555")
def cloud(x, y, w, h, fc, ec):
    for cx, cy, r in [(0.25, 0.45, 0.33), (0.5, 0.6, 0.4), (0.75, 0.45, 0.33), (0.5, 0.35, 0.36)]: ax.add_patch(Ellipse((x + cx * w, y + cy * h), r * w * 0.9, r * h * 1.3, fc=fc, ec=ec, lw=2))
    ax.add_patch(Ellipse((x + 0.5 * w, y + 0.45 * h), 0.78 * w, 0.62 * h, fc=fc, ec="none"))
def node(x, y, sigma, filled, color, label=None, r=2.3):
    ax.add_patch(Circle((x, y), r, fc=color if filled else "white", ec=color if filled else GRY, lw=2.2, ls="-" if filled else (0, (2, 2))))
    text(x, y - r - 2.4, "σ %.2f" % sigma, 9, c=INK if filled else "#9aa0a8")
    if label: text(x, y + r + 2.2, label, 10, c=color, w="bold")
# ---------------- device ----------------
box(4, 26, 72, 52, "#eef8f8", DEV, lw=2.5, r=4); text(10, 74.5, "on device", 16, DEV, "bold", ha="left")
ax.add_patch(Rectangle((60, 69), 12, 7.5, fc="white", ec=DEV, lw=2)); ax.add_patch(Rectangle((61.3, 70.2), 9.4, 5.1, fc=DEV, alpha=0.25, ec="none")); text(66, 72.7, "FP8 14B", 8.5, DEV, "bold")
# noise -> few steps -> low-res draft
noisy_grid(8, 48, 12, 7, 5, 9, 1, DEV, 1.0); text(14, 46, "noise", 9.5, "#555")
xs = [27, 37, 47, 57]; sig = [1.0, 0.94, 0.83, 0.63]; y0 = 51.5
for i, (x, s) in enumerate(zip(xs, sig)): node(x, y0, s, i in (0, 2), DEV)
arrow(20.5, y0, xs[0] - 2.6, y0, DEV); arrow(xs[0] + 2.6, y0, xs[2] - 2.6, y0, DEV, cs="arc3,rad=-0.35")   # skip the second node
text(37, 58.5, "2 of 4 steps", 11, DEV, "bold")
noisy_grid(63, 47.5, 10.5, 6.3, 4, 7, 2, DEV, 0.25); text(68.2, 45.3, "low-res draft", 9.5, "#555")
arrow(xs[2] + 2.6, y0, 62.5, y0, DEV)
# show
display(10, 29.5, 20, 11, DEV, "shown", "draft stream", "25 fps", lx=8)
arrow(66, 47, 66, 44.2, DEV, style="-"); arrow(66, 44.2, 24, 44.2, DEV, style="-"); arrow(24, 44.2, 24, 41, DEV)
# verifier head
ax.add_patch(Polygon([[47, 41.5], [52, 36.5], [47, 31.5], [42, 36.5]], fc="white", ec=VER, lw=2.5)); text(47, 36.5, "verifier", 9.5, VER, "bold")
arrow(xs[2], y0 - 2.6, 47, 42, VER, lw=2, ls=(0, (3, 2)), ms=14); text(39.5, 44.5, "hidden\nfeatures", 8.5, VER)
text(47, 29.0, "✓ accept      ✗ send", 10, VER, "bold")
arrow(52.5, 36.5, 86, 36.5, VER, lw=2.4); text(69, 38.6, "flagged chunks only  ·  latent, 0.4 MB", 9.5, VER)
# ---------------- cloud ----------------
cloud(84, 31, 72, 50, "#eef0fd", CLD); text(120, 76.5, "cloud", 16, CLD, "bold")
ax.add_patch(Rectangle((138, 68.5), 12, 7.5, fc="white", ec=CLD, lw=2)); ax.add_patch(Rectangle((139.3, 69.7), 9.4, 5.1, fc=CLD, alpha=0.25, ec="none")); text(144, 72.2, "FP8 14B", 8.5, CLD, "bold")
noisy_grid(92, 48.5, 10.5, 6.3, 4, 7, 2, DEV, 0.25); arrow(86.5, 36.5, 97, 47.8, VER, lw=2.4, cs="arc3,rad=-0.25")
arrow(103.5, 51.5, 109, 51.5, CLD); noisy_grid(109.5, 47.3, 13, 8, 6, 11, 3, CLD, 0.18); text(116, 45, "upsample", 9.5, "#555")
node(132, 51.5, 0.63, True, CLD, "+1 step"); arrow(123, 51.5, 129.4, 51.5, CLD)
grid(140, 47.3, 13, 8, 6, 11, CLD, lw=0.6); text(146.5, 45, "full-res chunk", 9.5, "#555"); arrow(134.6, 51.5, 139.5, 51.5, CLD)
# back to the device: corrected stream
display(112, 22, 22, 11, CLD, "corrected stream", "0.45 s behind", "25 fps")
arrow(146.5, 47, 146.5, 27.5, CLD, style="-"); arrow(146.5, 27.5, 134.5, 27.5, CLD)
# ---------------- timeline ----------------
text(6, 20.5, "chunk pipeline (0.75 s of video each)", 11, INK, "bold", ha="left")
tx0, cw = 30, 15
for k in range(7):
    x = tx0 + k * cw; box(x, 12.5, cw - 1, 5, DEV, DEV, lw=0, r=1); text(x + cw / 2 - 0.5, 15, "draft c+%d" % k if k else "draft c", 9, "white", "bold")
flag = [1, 0, 1, 1, 0, 0, 1]
for k in range(7):
    x = tx0 + k * cw
    if flag[k]: box(x + 2.5, 4.5, cw * 0.92 - 1, 5, CLD, CLD, lw=0, r=1); text(x + 2.5 + cw * 0.46 - 0.5, 7, "+1 step", 9, "white", "bold"); arrow(x + cw / 2 - 0.5, 12.5, x + cw / 2 + 2, 9.5, VER, lw=1.8, ms=12)
    else: text(x + cw / 2 - 0.5, 7, "✓", 13, VER, "bold")
text(tx0 - 2, 15, "device", 10, DEV, "bold", ha="right"); text(tx0 - 2, 7, "cloud", 10, CLD, "bold", ha="right")
arrow(tx0, 2, tx0 + 7 * cw, 2, INK, lw=1.2, ms=10); text(tx0 + 7 * cw + 1, 2, "time", 9, INK, ha="left")
fig.savefig(out, facecolor="white"); print("saved", out)
