"""Final comparison video (10/12): per prompt a 2x5 grid of 10 arms, seed 1234, same start image + camera path:
row 1 = 14B 4-step teacher | 14B 2-step 832 | ours 832 2-step fused | ours 704 fused (no refiner)
row 2 = ours 672 fused (no refiner) | student LingBot 1.3B | Matrix-Game 2.0 | Waypoint-1.5-1B.
Each tile carries name, resolution, measured fps and that clip's VLM drift / quality. 12 prompts, one after another.
usage: python scripts/cmp_final_video.py [--out paper/1012/videos/cmp_final.mp4] [--tasks a,b]"""
import argparse, os, json, subprocess, cv2, numpy as np
S = "/home/hk28965/specwm"; O = S + "/lingbot-world-v2/specwm_out"; P = S + "/paper/1012"
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=P + "/videos/cmp_final.mp4")
ap.add_argument("--tasks", default="easy_lake,easy_greatwall,easy_stonehenge_sunset,easy_urban_rain,easy_dragon_night,pouring_water_into_cup,domino_falling,hourglass_accumulation,newtons_cradle,billiard_break,sand_filling_up_jar,sandcastle_wave_impact"); a = ap.parse_args()
CW, CH, LAB, COLS = 352, 203, 40, 5
ARMS = [("14B 4-step teacher", "832x480", 4.9, lambda t: "cmp10/%s/uniform4_s1234/video.mp4" % t),
        ("14B 2-step", "832x480", 8.3, lambda t: "cmp10/%s/uniform2_s1234/video.mp4" % t),
        ("ours: 832 2-step fused", "832x480", 11.8, lambda t: "fused/%s/f832_n0_500_s1234/video.mp4" % t),
        ("ours: 832 fused + partial commit", "832x480", 9.7, lambda t: "fused/%s/pc1_832_s1234/cloud/spec.mp4" % t),
        ("ours: 704 fused + partial commit", "704x384", 15.8, lambda t: "fused/%s/pc1_704_s1234/cloud/spec.mp4" % t),
        ("ours: 704 fused", "704x384", 18.7, lambda t: "fused/%s/fc704_s1234/cloud/spec.mp4" % t),
        ("ours: 672 fused", "672x352", 22.4, lambda t: "fused/%s/fc672_s1234/cloud/spec.mp4" % t),
        ("student: LingBot 1.3B 4-step", "832x480", 10.4, lambda t: "cmp10/%s/draft_1p3b_s1234/video.mp4" % t),
        ("Matrix-Game 2.0", "640x352", 20.5, lambda t: "mg2/seeds/%s_s1234/video.mp4" % t),
        ("Waypoint-1.5-1B", "1024x512", 28.3, lambda t: "waypoint/seeds/%s_r1/output.mp4" % t)]
SC = {}
for f in ["vlm_scores_3seed.json", "vlm_scores_fused.json", "vlm_scores_diag.json"]:
    for k, v in json.load(open(P + "/" + f)).items(): SC[k.split("specwm_out/")[-1]] = (v["drift"], v["quality"])
def frames(p):
    cap = cv2.VideoCapture(p); out = []
    while True:
        ok, f = cap.read()
        if not ok: break
        h, w = f.shape[:2]; s = min(CW / w, CH / h); f = cv2.resize(f, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        c = np.zeros((CH, CW, 3), np.uint8); y, x = (CH - f.shape[0]) // 2, (CW - f.shape[1]) // 2; c[y:y + f.shape[0], x:x + f.shape[1]] = f; out.append(c)
    return out
def label(name, res, fps, sc, hl):
    L = np.full((LAB, CW, 3), 24, np.uint8); col = (90, 200, 255) if hl else (235, 235, 235)
    cv2.putText(L, name, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    s2 = "%s | %.1f fps" % (res, fps) + (" | VLM drift %d, q %d" % sc if sc else "")
    cv2.putText(L, s2, (5, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1, cv2.LINE_AA); return L
os.makedirs(os.path.dirname(a.out), exist_ok=True); tmp = a.out.replace(".mp4", "_raw.mp4"); W, H = COLS * CW, 2 * (CH + LAB) + 30
vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 16, (W, H))
for t in a.tasks.split(","):
    tiles = []
    for name, res, fps, f in ARMS:
        p = f(t); fr = frames(O + "/" + p) if os.path.exists(O + "/" + p) else []
        if not fr: print("missing", t, name); fr = [np.zeros((CH, CW, 3), np.uint8)]
        tiles.append((np.vstack, fr, label(name, res, fps, SC.get(p), name == "ours: 704 fused + partial commit")))
    n = max(len(fr) for _, fr, _ in tiles)
    head = np.full((30, W, 3), 0, np.uint8); cv2.putText(head, "%s  (seed 1234, same start image + camera path)" % t, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    for i in range(n):
        cells = [np.vstack([lab, fr[min(i, len(fr) - 1)]]) for _, fr, lab in tiles]
        vw.write(np.vstack([head, np.hstack(cells[:COLS]), np.hstack(cells[COLS:])]))
    print("done", t, n, flush=True)
vw.release(); subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", a.out], check=True); os.remove(tmp); print("[OUT]", a.out)
