"""Frame-jump probe: consecutive-frame mean abs difference (grey, 208 px wide) per video, folded by phase mod 12 (one chunk = 12 frames).
A chunk-boundary jump shows as one phase whose mean diff is far above the others. usage: python scripts/jump_probe.py LABEL=GLOB ..."""
import sys, glob, cv2, numpy as np
def diffs(p):
    cap = cv2.VideoCapture(p); prev = None; d = []
    while True:
        ok, f = cap.read()
        if not ok: break
        g = cv2.cvtColor(cv2.resize(f, (208, int(208 * f.shape[0] / f.shape[1]))), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None: d.append(np.abs(g - prev).mean())
        prev = g
    return np.array(d)
print("| arm | clips | frames | phase of max | max-phase / median-phase ↓ | top-5 single-step jumps / median step |")
print("|---|---|---|---|---|---|")
for arg in sys.argv[1:]:
    lab, g = arg.split("=", 1); fs = sorted(glob.glob(g)); P = []; tops = []; n = 0
    for f in fs:
        d = diffs(f); n = len(d) + 1
        if len(d) < 40: continue
        ph = np.array([d[k::12][1:].mean() for k in range(12)]); P.append(ph / np.median(ph))
        tops.append(np.sort(d)[-5:].mean() / np.median(d))
    P = np.array(P).mean(0); k = int(P.argmax())
    print("| %s | %d | %d | %d | %.2f | %.1f |" % (lab, len(fs), n, k, P[k], np.mean(tops)), flush=True)
    print("   phases:", " ".join("%.2f" % x for x in P))
