import os
import numpy as np, cv2, os, imageio
S = os.environ["SCRATCH"]+""
def read(p, step=1):
    c = cv2.VideoCapture(p); fr = []
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return np.stack(fr[::step])
# 1) 추측 디코딩 draft vs target 프레임
D = read(S + "/tmp_draft.mp4"); T = read(S + "/tmp_target.mp4"); P = read(S + "/tmp_spec.mp4")
n = min(len(D), len(T), len(P)); print("draft/target/spec 프레임:", len(D), len(T), len(P))
np.savez_compressed(S + "/spec_frames.npz", draft=D[:n], target=T[:n], spec=P[:n])
# 2) 손 조작 3분할 영상 (FAST | REFINE | BASE), 원본 16fps 81프레임
R = S + "/lingbot-world-v2/specwm_out"
F = read(R + "/hybrid/hands/FAST_asis/video.mp4")
Rf = read(R + "/hybrid/hands/REFINE_0.40_0.20_0.08/video.mp4")
B = read(R + "/hands/base_p1/video.mp4")
n = min(len(F), len(Rf), len(B)); print("hands 프레임:", len(F), len(Rf), len(B))
def label(img, txt):
    img = img.copy(); cv2.rectangle(img, (0, 0), (300, 34), (0, 0, 0), -1)
    cv2.putText(img, txt, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2); return img
frames = [np.concatenate([label(F[i], "FAST (18s)"), label(Rf[i], "FAST + Base refine (47s)"),
                          label(B[i], "BASE (816s)")], axis=1) for i in range(n)]
imageio.mimwrite(S + "/hands_3up.mp4", frames, fps=16, quality=8)
print("hands_3up.mp4 저장:", frames[0].shape)
