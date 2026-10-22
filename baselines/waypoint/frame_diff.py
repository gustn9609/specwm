"""Compare two clips frame by frame (decoded with OpenCV): mean / max per-frame mean-abs-diff (0-255) and PSNR; for the batch-vs-single determinism check.
usage: envrun.sh python frame_diff.py A.mp4 B.mp4"""
import sys, cv2, numpy as np
def frames(p):
    c = cv2.VideoCapture(p); out = []
    while True:
        ok, f = c.read()
        if not ok: break
        out.append(f.astype(np.float32))
    return out
A, B = frames(sys.argv[1]), frames(sys.argv[2]); n = min(len(A), len(B))
mad = np.array([np.abs(A[i] - B[i]).mean() for i in range(n)]); mse = np.array([((A[i] - B[i]) ** 2).mean() for i in range(n)])
psnr = 10 * np.log10(255 ** 2 / np.maximum(mse, 1e-6))
print("[frame_diff] %s vs %s: frames %d/%d, mean-abs-diff mean %.3f max %.3f (frame %d), PSNR mean %.1f dB min %.1f dB; first 4 frames MAD %s; last 4 MAD %s" % (
    sys.argv[1], sys.argv[2], len(A), len(B), mad.mean(), mad.max(), int(mad.argmax()), psnr.mean(), psnr.min(), np.round(mad[:4], 3).tolist(), np.round(mad[-4:], 3).tolist()))
