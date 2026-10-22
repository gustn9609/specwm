"""Chunk-boundary jump: for each boundary b (frame 8|9, then every 12 frames) r = d[b] / mean(d[b-2], d[b-1], d[b+1], d[b+2]),
d = consecutive-frame mean abs difference (grey, 208 px). Reports mean r over boundaries and clips, the share of boundaries with r > 1.5,
and the same ratio at the latent-internal positions (b+4, b+8) as a control. usage: python scripts/boundary_jump.py LABEL=GLOB ..."""
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
def ratios(d, off):
    r = []
    for b in range(8 + off, len(d) - 2, 12):
        if b - 2 < 0: continue
        r.append(d[b] / max(1e-6, np.mean([d[b - 2], d[b - 1], d[b + 1], d[b + 2]])))
    return r
print("| arm | clips | boundary ratio ↓ | boundaries > 1.5× ↓ | control (mid-chunk) ratio |"); print("|---|---|---|---|---|")
for arg in sys.argv[1:]:
    lab, g = arg.split("=", 1); R = []; C = []
    for f in sorted(glob.glob(g)):
        d = diffs(f)
        if len(d) < 40: continue
        R += ratios(d, 0); C += ratios(d, 4) + ratios(d, 8)
    R = np.array(R); C = np.array(C)
    print("| %s | %d | %.2f | %.0f %% | %.2f |" % (lab, len(glob.glob(g)), R.mean(), 100 * (R > 1.5).mean(), C.mean()), flush=True)
