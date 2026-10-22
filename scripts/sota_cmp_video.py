"""SOTA comparison video: per prompt a 1x4 row [14B 4-step teacher | ours 576x320 2-step + SR | Waypoint-1.5-1B | Matrix-Game 2.0], same start image,
same camera trajectory (converted to each model's action space), 153 frames each; labels carry resolution and generation time. Prints the timing table.
usage: python scripts/sota_cmp_video.py [--out paper/1012/videos/sota_cmp10.mp4]"""
import argparse, os, json, subprocess, numpy as np, cv2
S = "/home/hk28965/specwm"; O = S + "/lingbot-world-v2/specwm_out"
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=S + "/paper/1012/videos/sota_cmp10.mp4"); a = ap.parse_args()
T = ["easy_lake", "easy_greatwall", "easy_stonehenge_sunset", "easy_urban_rain", "easy_dragon_night", "pouring_water_into_cup", "domino_falling", "hourglass_accumulation", "newtons_cradle", "billiard_break"]
CW, CH, LAB = 384, 216, 24
def tm_ours(t, arm):
    p = O + "/cmp10/%s/%s/loop.json" % (t, arm); d = json.load(open(p)) if os.path.exists(p) else {}; return d.get("gen_s")
ARMS = [("14B 4-step teacher, 832x480 (ours' reference)", lambda t: O + "/cmp10/%s/uniform4_s1234/video.mp4" % t, lambda t: tm_ours(t, "uniform4_s1234")),
        ("ours: 14B 2-step 576x320 + SR", lambda t: O + "/cmp10_lr_sr/%s/lr576_u2_s1234/video.mp4" % t, lambda t: tm_ours(t, "lr576_u2_s1234")),
        ("Waypoint-1.5-1B, 1024x512 (keyboard/mouse)", lambda t: O + "/waypoint/cmp10/%s/output.mp4" % t, lambda t: json.load(open(O + "/waypoint/cmp10/%s/timing.json" % t))["gen_s"] if os.path.exists(O + "/waypoint/cmp10/%s/timing.json" % t) else None),
        ("Matrix-Game 2.0, 640x352 (keyboard/mouse)", lambda t: O + "/mg2/cmp10/%s/video.mp4" % t, lambda t: json.load(open(O + "/mg2/cmp10/%s/timing.json" % t))["gen_s"] if os.path.exists(O + "/mg2/cmp10/%s/timing.json" % t) else None),
        ("Matrix-Game 3.5 distilled, 1280x704 (image+camera path+text)", lambda t: O + "/mg35/cmp10/%s/result.mp4" % t, lambda t: mg35_time(t))]
def mg35_time(t):
    p = S + "/exp/queue/mg35_%s.log" % t
    if t == "easy_lake": p = S + "/exp/queue/mg35_test.log"
    if not os.path.exists(p): return None
    import re; m = re.findall(r"inference_s ([0-9.]+)", open(p, errors="ignore").read()); return float(m[-1]) if m else None
def frames(p):
    cap = cv2.VideoCapture(p); out = []
    while True:
        ok, f = cap.read()
        if not ok: break
        out.append(cv2.resize(f, (CW, CH)))
    return out
os.makedirs(os.path.dirname(a.out), exist_ok=True); tmp = a.out.replace(".mp4", "_raw.mp4"); W, H = len(ARMS) * CW, CH + LAB + 26
vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 16, (W, H)); table = {}
for t in T:
    V = []; TM = []
    for lab, vp, tf in ARMS:
        p = vp(t); V.append(frames(p) if p and os.path.exists(p) else []); TM.append(tf(t))
    n = max(len(v) for v in V)
    if n == 0: continue
    for i in range(n):
        canvas = np.zeros((H, W, 3), np.uint8); cv2.putText(canvas, "%s  (same start image + same camera path, frame %d/%d)" % (t, i + 1, n), (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        for k, (lab, _, _) in enumerate(ARMS):
            x0 = k * CW; f = V[k][min(i, len(V[k]) - 1)] if V[k] else np.full((CH, CW, 3), 40, np.uint8); canvas[26 + LAB:26 + LAB + CH, x0:x0 + CW] = f
            txt = lab + ("  gen %.1fs" % TM[k] if TM[k] else ""); cv2.putText(canvas, txt, (x0 + 4, 26 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255) if lab.startswith("ours") else (255, 255, 255), 1)
        vw.write(canvas)
    table[t] = TM
vw.release(); subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out]); os.remove(tmp); print("wrote", a.out)
print("| model | gen s for 153 frames ↓ (median over prompts) | generated fps ↑ | " + " | ".join(t.replace("_", " ") for t in table) + " |"); print("|" + "---|" * (3 + len(table)))
for k, (lab, _, _) in enumerate(ARMS):
    L = [table[t][k] for t in table if table[t][k]]
    if not L: continue
    m = float(np.median(L)); print("| %s | %.1f | %.1f | " % (lab, m, 153 / m) + " | ".join(("%.1f" % table[t][k]) if table[t][k] else "–" for t in table) + " |")
