"""Side-by-side comparison video: for each prompt a 2x4 grid of the 8 arms (label + generation time burnt in), prompts concatenated.
Also prints the timing table. usage: python scripts/cmp10_video.py [--out paper/1012/videos/cmp10.mp4]"""
import argparse, os, json, glob, subprocess, numpy as np, cv2
S = "/home/hk28965/specwm"; O = S + "/lingbot-world-v2/specwm_out/cmp10"
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=S + "/paper/1012/videos/cmp10.mp4"); ap.add_argument("--cell", default="416x240"); a = ap.parse_args()
CW, CH = [int(v) for v in a.cell.split("x")]; LAB = 22
T = ["easy_lake", "easy_greatwall", "easy_stonehenge_sunset", "easy_urban_rain", "easy_dragon_night", "pouring_water_into_cup", "domino_falling", "hourglass_accumulation", "newtons_cradle", "billiard_break"]
ARMS = [("14B 4-step (teacher)", "uniform4_s1234", "gen"), ("14B 3-step", "uniform3_s1234", "gen"), ("14B 2-step", "uniform2_s1234", "gen"), ("ours: token early exit", "early0.24_flash_s1234", "gen"),
        ("ours: chunk-wise adaptive", "diag_learned_dense_0.24_chunk0.7_s1234", "gen_deploy"), ("ours: anchor split (2 GPU)", "anchor_d2_s1234", "gen"), ("ours: 576x320 2-step + SR", "lr576_u2_s1234", "gen"), ("1.3B draft", "draft_1p3b_s1234", "gen")]
def vid_path(t, arm):
    if arm.startswith("lr576"): p = O + "_lr_sr/%s/%s/video.mp4" % (t, arm)
    else: p = O + "/%s/%s/video.mp4" % (t, arm)
    if not os.path.exists(p) and os.path.exists(os.path.dirname(p) + "/spec.mp4"): p = os.path.dirname(p) + "/spec.mp4"
    return p if os.path.exists(p) else None
def timing(t, arm, kind):
    lj = O + "/%s/%s/loop.json" % (t, arm)
    if not os.path.exists(lj): return None
    d = json.load(open(lj)); tm = d.get("times", {})
    if kind == "gen_deploy": return dict(gen=tm.get("denoise", 0) + tm.get("commit", 0), e2e=d.get("e2e_s", 0) - tm.get("verify", 0), steps=d.get("mean_steps"))   # diag mode: drop the dense-reference recompute
    return dict(gen=d.get("gen_s"), e2e=d.get("e2e_s"), steps=d.get("mean_steps"))
def frames(p):
    cap = cv2.VideoCapture(p); out = []
    while True:
        ok, f = cap.read()
        if not ok: break
        out.append(cv2.resize(f, (CW, CH)))
    return out
os.makedirs(os.path.dirname(a.out), exist_ok=True); tmp = a.out.replace(".mp4", "_raw.mp4"); W, H = 4 * CW, 2 * (CH + LAB) + 26
vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 16, (W, H)); rows = {}
for t in T:
    V = []; TM = []
    for lab, arm, kind in ARMS:
        p = vid_path(t, arm); V.append(frames(p) if p else []); TM.append(timing(t, arm, kind))
    n = max(len(v) for v in V)
    if n == 0: continue
    for i in range(n):
        canvas = np.zeros((H, W, 3), np.uint8); cv2.putText(canvas, "%s   (seed 1234, frame %d/%d)" % (t, i + 1, n), (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        for k, (lab, arm, kind) in enumerate(ARMS):
            r, c = divmod(k, 4); y0 = 26 + r * (CH + LAB); x0 = c * CW
            f = V[k][min(i, len(V[k]) - 1)] if V[k] else np.full((CH, CW, 3), 40, np.uint8); canvas[y0 + LAB:y0 + LAB + CH, x0:x0 + CW] = f
            tm = TM[k]; txt = "%s  %s" % (lab, ("gen %.1fs" % tm["gen"]) if tm and tm.get("gen") else "")
            cv2.putText(canvas, txt, (x0 + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255) if lab.startswith("ours") else (255, 255, 255), 1)
        vw.write(canvas)
    rows[t] = TM
vw.release(); subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out]); os.remove(tmp); print("wrote", a.out)
print("| arm | mean steps | gen s ↓ (mean over %d prompts) | e2e s ↓ | speed-up vs 4-step (gen) | " % len(rows) + " | ".join(t.replace("_", " ") for t in rows) + " |"); print("|" + "---|" * (5 + len(rows)))
g4 = np.mean([rows[t][0]["gen"] for t in rows if rows[t][0]])
for k, (lab, arm, kind) in enumerate(ARMS):
    L = [rows[t][k] for t in rows if rows[t][k]]
    if not L: print("| %s | – | – | – | – |" % lab); continue
    g = np.mean([x["gen"] for x in L]); print("| %s | %s | %.1f | %.1f | %.2f× | " % (lab, ("%.2f" % np.mean([x["steps"] for x in L if x["steps"]])) if any(x["steps"] for x in L) else "–", g, np.mean([x["e2e"] for x in L]), g4 / g) + " | ".join(("%.1f" % rows[t][k]["gen"]) if rows[t][k] else "–" for t in rows) + " |")
