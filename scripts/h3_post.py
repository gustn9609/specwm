#!/usr/bin/env python3
"""H3 mp4 (24fps, 768p) -> 우리 분석 형식: frames_all.npy (4fps, 464x832 리사이즈) 를 각 h3/<name>/ 에 생성 (VR/CLIP/DINO 도구 공용)."""
import os, glob, numpy as np, cv2
S = os.environ.get("SCRATCH", "/home/hk28965/specwm"); R = S + "/lingbot-world-v2/specwm_out/h3"
for v in sorted(glob.glob(R + "/*/video.mp4")):
    d = os.path.dirname(v); out = os.path.join(d, "frames_all.npy")
    if os.path.exists(out): continue
    c = cv2.VideoCapture(v); fps = c.get(cv2.CAP_PROP_FPS) or 24; step = max(1, int(round(fps / 4))); fr = []; i = 0
    while True:
        ok, f = c.read()
        if not ok: break
        if i % step == 0: fr.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (832, 464), interpolation=cv2.INTER_AREA))
        i += 1
    np.save(out, np.stack(fr).astype("uint8")); print(os.path.basename(d), "fps", fps, "frames", i, "->", len(fr), flush=True)
print("[H3_POST_DONE]")
