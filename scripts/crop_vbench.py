"""재접지 arm 의 '생성 구간만' 잘라 VBench 용 mp4 를 만든다 (Base 프레임 혼입 제거).
resync_k: 프레임 12k 이후 / keep_k3_L: 12(3+L) 이후 / fast_alone, ctrl: 12*6 이후(공정 비교용 동일 창) / every3: 전체 (전부 생성)
+ 통일 post-window(3청크=36프레임) 버전도 저장.
"""
import os; os.environ["OMP_NUM_THREADS"] = "1"
import cv2; cv2.setNumThreads(1)
import numpy as np, imageio, json, glob
S = os.environ["SCRATCH"]+""; RV = S + "/lingbot-world-v2/specwm_out/resync_v2"; OUT = S + "/vbench_crops"; os.makedirs(OUT, exist_ok=True)
rep = json.load(open(RV + "/resync_v2.json")); spec = []
def read(p):
    c = cv2.VideoCapture(p); fr = []
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return np.stack(fr)
for key, r in rep.items():
    if "plan" not in r: continue
    cfg, name = key.split("/"); plan = r["plan"]
    if plan.startswith("B") or plan.startswith("F"):
        k = len(plan) - len(plan.lstrip("BF"))
    else:
        k = 6 if not name.startswith("every3") else 0
    p = RV + "/" + key + "/video.mp4"
    if not os.path.exists(p): continue
    V = read(p); a = 12 * k
    tag = key.replace("/", "__")
    imageio.mimwrite(OUT + "/%s__gen.mp4" % tag, list(V[a:]), fps=16, quality=8, codec="libx264", pixelformat="yuv420p")
    spec.append(("%s__gen" % tag, OUT + "/%s__gen.mp4" % tag))
    if a + 36 <= len(V):
        imageio.mimwrite(OUT + "/%s__post3.mp4" % tag, list(V[a:a+36]), fps=16, quality=8, codec="libx264", pixelformat="yuv420p")
        spec.append(("%s__post3" % tag, OUT + "/%s__post3.mp4" % tag))
open(OUT + "/videos.txt", "w").write("\n".join("%s=%s" % kv for kv in spec)); print("crops:", len(spec))
