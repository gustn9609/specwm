import os; os.environ["OMP_NUM_THREADS"] = "1"
import cv2; cv2.setNumThreads(1); import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = os.environ["SCRATCH"]+""; R = S + "/lingbot-world-v2/specwm_out"
def read(p):
    c = cv2.VideoCapture(p); fr = []
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return np.stack(fr)
V = {"A Fast alone": read(R + "/resync_v2/W18_sink6/fast_alone_s1234/video.mp4"), "B Base16 independent": read(R + "/revisit/base_official_S05_s16/video.mp4"),
     "C mask+Fast prefix": read(R + "/revisit/base_prefix_S05_L9/video.mp4"), "D laundered+RePaint": read(R + "/revisit/e3c_repaint_laundered/video.mp4"), "B70 Base70 official": read(R + "/revisit/base_official_S05/video.mp4")}
print({k: v.shape for k, v in V.items()})
print("frame-to-frame |diff| frames 28..38 (boundary 32->33)")
for k, v in V.items():
    d = np.abs(v[1:].astype(np.float32) - v[:-1].astype(np.float32)).mean(axis=(1, 2, 3)); med = np.median(d)
    print("%-22s median %5.2f | %s | 32->33 = %.2fx median, max/median over clip %.2fx at f%d" % (k, med, " ".join("%5.1f" % d[i] for i in range(28, 38)), d[32] / med, d.max() / med, int(d.argmax())))
fig, axes = plt.subplots(2, 8, figsize=(2.3*8, 3.0)); fig.patch.set_facecolor("white")
for r, k in enumerate(["C mask+Fast prefix", "D laundered+RePaint"]):
    for c, i in enumerate(range(29, 37)):
        ax = axes[r, c]; ax.imshow(V[k][i][::2, ::2]); ax.set_xticks([]); ax.set_yticks([]); ax.set_title("f%d %.2fs%s" % (i, i/16, " prefix" if i <= 32 else " GEN"), fontsize=8)
        if c == 0: ax.set_ylabel(k, fontsize=8)
fig.suptitle("Boundary: last prefix frame f32 (2.00 s) -> first generated f33", fontsize=10, fontweight="bold"); fig.tight_layout(); fig.savefig(S + "/boundary_CD.png", dpi=115, bbox_inches="tight"); plt.close()
ks = list(range(0, 161, 16))
fig, axes = plt.subplots(3, len(ks), figsize=(2.2*len(ks), 4.6)); fig.patch.set_facecolor("white")
for r, k in enumerate(["A Fast alone", "B Base16 independent", "B70 Base70 official"]):
    for c, i in enumerate(ks):
        ax = axes[r, c]; ax.imshow(V[k][min(i, len(V[k])-1)][::2, ::2]); ax.set_xticks([]); ax.set_yticks([]); ax.set_title("t=%d s" % (i//16), fontsize=8)
        if c == 0: ax.set_ylabel(k, fontsize=8)
fig.tight_layout(); fig.savefig(S + "/A_vs_B.png", dpi=115, bbox_inches="tight")
# 선명도(라플라시안 분산) 와 밝기
print("sharpness (laplacian var, mean over frames) / brightness")
for k, v in V.items():
    sh = np.mean([cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_32F).var() for f in v[::8]]); print("%-22s sharp %.0f  bright %.1f" % (k, sh, v.mean()))
