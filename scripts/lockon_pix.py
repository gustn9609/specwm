import os; os.environ["OMP_NUM_THREADS"] = "1"
import cv2; cv2.setNumThreads(1); import numpy as np, json
S = os.environ["SCRATCH"]+""; R = S + "/lingbot-world-v2/specwm_out"
def read(p):
    c = cv2.VideoCapture(p); fr = []
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return np.stack(fr)
FA = read(R + "/resync_v2/W18_sink6/fast_alone_s1234/video.mp4").astype(np.float32); B16 = read(R + "/revisit/base_official_S05_s16/video.mp4").astype(np.float32)
med = np.median(np.abs(FA[1:] - FA[:-1]).mean(axis=(1, 2, 3)))
provs = {"raw": R + "/resync_v2/W18_sink6/fast_alone_s1234/video.mp4", "l15": R + "/lockon/launder/REFINE_0.15_0.07_0.03/video.mp4", "l25": R + "/lockon/launder/REFINE_0.25_0.12_0.05/video.mp4", "l40": R + "/hybrid/s05_launder/REFINE_0.40_0.20_0.08/video.mp4", "l60": R + "/lockon/launder/REFINE_0.60_0.30_0.12/video.mp4", "l80": R + "/lockon/launder/REFINE_0.80_0.40_0.16/video.mp4"}
out = {}
print("%-5s %10s %9s %9s %10s %9s" % ("prov", "gap(pix)", "snap5>6", "freeze", "sharp6-12", "bright"))
for n, p in provs.items():
    P = read(p).astype(np.float32); X = read(R + "/lockon/W18_sink6/kvrw_%s_s1234/video.mp4" % n).astype(np.float32)
    gap = float(np.abs(FA[:72] - P[:72]).mean())                                   # 화면(Fast) vs KV(provider) 0-4.5 s 픽셀 격차
    d = np.abs(X[1:] - X[:-1]).mean(axis=(1, 2, 3)); snap = float(np.abs(FA[71] - X[72]).mean() / med)   # 표시 Fast 마지막(청크5) -> 생성 첫 프레임(청크6)
    freeze = float(d[72:78].mean() / med); sh = float(np.mean([cv2.Laplacian(cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_32F).var() for f in X[72::8]]))
    out[n] = dict(gap_pix=round(gap, 2), snap=round(snap, 2), freeze=round(freeze, 2), sharp=round(sh), bright=round(float(X[72:].mean()), 1)); print("%-5s %10.2f %9.2f %9.2f %10.0f %9.1f" % (n, gap, snap, freeze, sh, X[72:].mean()))
json.dump(out, open(R + "/lockon/pix_metrics.json", "w"), indent=1)
