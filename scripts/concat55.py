import os; os.environ["OMP_NUM_THREADS"] = "1"
import cv2; cv2.setNumThreads(1); import numpy as np, imageio
S = os.environ["SCRATCH"]+""; R = S + "/lingbot-world-v2/specwm_out/win60"; OUT = S + "/compare_videos"
def label(img, txt):
    img = img.copy(); (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2); cv2.rectangle(img, (0, 0), (w + 16, h + 16), (0, 0, 0), -1); cv2.putText(img, txt, (8, h + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2); return img
vids = [(l, np.load(R + "/%s/frames_all.npy" % t)) for l, t in [("Fast window 4.5 s (55 s clip)", "W18"), ("window 9 s", "W36"), ("window 18 s", "W72")]]
T = max(len(v) for _, v in vids); frames = []
for t in range(T):
    cells = [label(cv2.resize(v[min(t, len(v)-1)], (532, 300), interpolation=cv2.INTER_AREA), l) for l, v in vids]; frames.append(np.concatenate(cells, axis=1))
imageio.mimwrite(OUT + "/12_window_sweep_55s.mp4", frames, fps=4, quality=7, codec="libx264", pixelformat="yuv420p"); print("12 저장", len(frames), "frames @4fps")
# 확인용 정지 프레임 2장
for name, t in [("06_resync_k3_k6_k9_every3", 100), ("15_e4_delayed_rollback_keep_xfade", 60)]:
    c = cv2.VideoCapture(OUT + "/%s.mp4" % name); c.set(cv2.CAP_PROP_POS_FRAMES, t); ok, f = c.read()
    if ok: cv2.imwrite(S + "/chk_%s.jpg" % name[:2], f, [cv2.IMWRITE_JPEG_QUALITY, 85])
