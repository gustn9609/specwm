import os; os.environ["OMP_NUM_THREADS"] = "1"
import cv2; cv2.setNumThreads(1); import imageio
S = os.environ["SCRATCH"]+""; R = S + "/lingbot-world-v2/specwm_out/epic"; O = S + "/epic/cut5"; os.makedirs(O, exist_ok=True)
def read(p, n):
    c = cv2.VideoCapture(p); fr = []
    while len(fr) < n:
        ok, f = c.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return fr
for W in ["G_peel_garlic", "F_cut_tomato"]:
    for tag, p in [("GT", S + "/epic/%s/gt_161f_16fps.mp4" % W), ("GEN_base70", R + "/%s_base70/video.mp4" % W), ("SIDEBYSIDE", S + "/compare_videos/18_epic_hands_%s_gt_vs_base.mp4" % W)]:
        fr = read(p, 81); imageio.mimwrite(O + "/%s_%s_5s.mp4" % (tag, W), fr, fps=16, quality=8, codec="libx264", pixelformat="yuv420p", output_params=["-threads", "2"]); print(tag, W, len(fr))
