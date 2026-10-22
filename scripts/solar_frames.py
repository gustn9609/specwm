import os; os.environ["OMP_NUM_THREADS"] = "1"
import cv2; cv2.setNumThreads(1)
import numpy as np, json
S = os.environ["SCRATCH"]+""
for name in ["video", "compare"]:
    c = cv2.VideoCapture(S + "/solar_stage2_s05_%s.mp4" % name); fr = []
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    V = np.stack(fr); print(name, V.shape, "fps", c.get(cv2.CAP_PROP_FPS))
    np.save(S + "/solar_stage2_s05_%s_frames.npy" % name, V[::4] if name == "video" else V[::8])
d = os.path.dirname([l for l in os.popen("find %s/solar_out/stage2_s05 -name manifest.json" % S).read().split()][0])
m = json.load(open(d + "/manifest.json")); print(json.dumps({k: (v if k != "artifacts" else "...") for k, v in m.items()}, indent=1)[:1200])
