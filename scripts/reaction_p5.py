"""[10/17 frozen baselines] Reaction test on control program P5 (PROTOCOL rule 7; specwm/4090_search.txt section 7): frames (and seconds) from each
EFFECTIVE input switch until the generated motion follows the new input. Motion signal = per-frame median RADIAL optical flow (expansion > 0 = camera
moving forward, contraction < 0 = backward, ~0 = still) plus the median HORIZONTAL flow (pan; content flowing left = camera panning right), from
cv2 Farneback on --width px grey frames; flow i is the motion from frame i-1 to frame i. P5 = fwd 0-2.5 s | still | back | fwd. The effective switch
times differ per model (encoding granularity; calib.json boundary notes): ours / MG2 / Astra 2.5, 5, 7.5 s; Waypoint 2.533, 5.0, 7.533 s;
ABot 2.75, 4.75, 7.75 s (its blocks take the net motion of their 12 frames, so the keys W W W - - S S S W W W switch at the block boundaries
33 / 57 / 93 decoded frames at 12 fps; calib.json['abot'].effective_switches; corrected 10/17 from 2.67/4.67/6.67+7.67).
Reaction at a switch = first frame >= the switch frame whose radial sign matches the new input (|v| > thr for a move, |v| < thr for still) and keeps
matching for --hold s; thr = --thr x the median |radial| over the steady part of the first forward segment (0.5 s .. first switch). A switch is only
VALID when the model was following the previous input during the --hold s before it (else a model that never moves would get 'reaction 0' at the
'still' switch); react = null when invalid or when no reaction occurs before the next switch. Per-clip JSON (keyed by the video path, label = arm):
{switches: [{t_s, sign, switch_frame, valid, react_frames, react_s}], n_valid, n_found, mean_react_frames, mean_react_s (over found), radial, horiz, ...}.
usage: /home/hk28965/specwm/scripts/envrun.sh python /home/hk28965/specwm/scripts/reaction_p5.py OUT.json LABEL=GLOB ... [--model auto|ours|waypoint|mg2|abot|astra]
       [--switches "2.5:0,5:-1,7.5:1"] [--thr 0.25 --hold 0.2 --width 320] [--redo]
       ... reaction_p5.py OUT.json --selftest   # synthetic zoom-in / still / zoom-out / zoom-in clip with known delays (3, 5, 2 frames) -> asserts the measurement
CPU only (no GPU)."""
import sys, os, glob, json, argparse, subprocess, numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("specs", nargs="*"); ap.add_argument("--model", default="auto", help="switch table: auto = from timing.json 'model' / the arm dir name")
ap.add_argument("--switches", default="", help='override the switch table: "t_s:sign,..." (sign +1 fwd, -1 back, 0 still)'); ap.add_argument("--thr", type=float, default=0.25, help="threshold = thr x median |radial| of the first forward segment")
ap.add_argument("--hold", type=float, default=0.2, help="s the new sign must persist (also the pre-switch validity window)"); ap.add_argument("--width", type=int, default=320); ap.add_argument("--redo", action="store_true")
ap.add_argument("--selftest", action="store_true"); ap.add_argument("--selftest-dir", default="/tmp/claude-1741758895/-home-hk28965/76e7a8ef-ae2e-455b-8c73-052f5bde2a30/scratchpad/reaction_selftest")
a = ap.parse_args()
SWITCHES = {"ours": [(2.5, 0), (5.0, -1), (7.5, 1)], "mg2": [(2.5, 0), (5.0, -1), (7.5, 1)], "astra": [(2.5, 0), (5.0, -1), (7.5, 1)],
            "waypoint": [(2.533, 0), (5.0, -1), (7.533, 1)], "abot": [(2.75, 0), (4.75, -1), (7.75, 1)]}   # [10/17 fix] ABot: block boundaries 33/57/93 frames @ 12 fps (calib.json['abot'].effective_switches.P5); the old 2.67/4.67/6.67(+7.67) table was wrong
MODEL_PAT = [("waypoint", "waypoint"), ("matrix-game", "mg2"), ("mg2", "mg2"), ("abot", "abot"), ("astra", "astra"), ("lingbot", "ours"), ("sf1p3b", "ours"), ("ours", "ours")]
def detect_model(path):
    d = os.path.dirname(os.path.abspath(path)); cands = []
    try: cands.append(str(json.load(open(os.path.join(d, "timing.json"))).get("model", "")).lower())
    except Exception: pass
    cands.append(os.path.basename(os.path.dirname(d)).lower())
    for c in cands:
        for pat, m in MODEL_PAT:
            if pat in c: return m
    return "ours"
def flows(path, width):
    """(radial [N], horiz [N], fps, N): per-frame median radial / horizontal Farneback flow in px/frame at `width` px (index 0 = 0)"""
    cap = cv2.VideoCapture(path); fps = float(cap.get(cv2.CAP_PROP_FPS)) or 16.0; prev = None; rad, hor = [], []
    while True:
        ok, f = cap.read()
        if not ok: break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); h = int(round(g.shape[0] * width / g.shape[1])); g = cv2.resize(g, (width, h), interpolation=cv2.INTER_AREA)
        if prev is None:
            yy, xx = np.mgrid[0:h, 0:width]; dx = xx - (width - 1) / 2.0; dy = yy - (h - 1) / 2.0; r = np.sqrt(dx ** 2 + dy ** 2)
            mask = r > 0.15 * min(h, width); ux = dx / np.maximum(r, 1e-6); uy = dy / np.maximum(r, 1e-6); rad.append(0.0); hor.append(0.0)
        else:
            fl = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            rad.append(float(np.median((fl[..., 0] * ux + fl[..., 1] * uy)[mask]))); hor.append(float(np.median(fl[..., 0])))
        prev = g
    cap.release(); return np.array(rad), np.array(hor), fps, len(rad)
def reaction(rad, fps, switches, thr_frac, hold_s):
    sw = sorted(switches); i0, i1 = int(round(0.5 * fps)), int(round(sw[0][0] * fps))
    ref = float(np.median(np.abs(rad[i0:i1]))) if i1 > i0 + 1 else float(np.median(np.abs(rad[1:]))); thr = thr_frac * ref; hold = max(2, int(round(hold_s * fps)))
    def ok_mask(sgn): return (np.abs(rad) < thr) if sgn == 0 else ((np.sign(rad) == sgn) & (np.abs(rad) > thr))
    out, prev_sign = [], 1   # P5 starts moving forward
    for k, (t, sgn) in enumerate(sw):
        f0 = int(round(t * fps)); f1 = int(round(sw[k + 1][0] * fps)) if k + 1 < len(sw) else len(rad); ok = ok_mask(sgn)
        valid = bool(f0 - hold >= 0 and ok_mask(prev_sign)[f0 - hold:f0].all()); react = None
        if valid:
            for f in range(f0, f1):
                if f + hold <= len(rad) and ok[f:f + hold].all(): react = f - f0; break
        out.append(dict(t_s=t, sign=sgn, prev_sign=prev_sign, switch_frame=f0, window_frames=f1 - f0, valid=valid, react_frames=react, react_s=(round(react / fps, 3) if react is not None else None))); prev_sign = sgn
    return out, thr, ref
def score(path, label, model=None):
    model = model or (a.model if a.model != "auto" else detect_model(path))
    sw = [(float(x.split(":")[0]), int(x.split(":")[1])) for x in a.switches.split(",")] if a.switches else SWITCHES[model]
    rad, hor, fps, N = flows(path, a.width); res, thr, ref = reaction(rad, fps, sw, a.thr, a.hold)
    found = [r["react_frames"] for r in res if r["react_frames"] is not None]
    return dict(label=label, model=model, fps=fps, frames=N, thr_px=round(thr, 4), ref_px=round(ref, 4), hold_frames=max(2, int(round(a.hold * fps))), switches=res, n_switches=len(res),
                n_valid=sum(r["valid"] for r in res), n_found=len(found), mean_react_frames=(float(np.mean(found)) if found else None), mean_react_s=(float(np.mean(found)) / fps if found else None),
                radial=[round(float(x), 3) for x in rad], horiz=[round(float(x), 3) for x in hor])
def selftest():
    """synthetic P5 clip: zoom in (fwd) / still / zoom out (back) / zoom in, with injected delays at the 3 switches; the measurement must recover them +-1 frame"""
    rng = np.random.default_rng(0); fps, W, H, n = 16, 320, 180, 161; delays = [3, 5, 2]; sw = SWITCHES["ours"]
    big = cv2.addWeighted(cv2.GaussianBlur(rng.integers(0, 256, (H * 4, W * 4, 3), dtype=np.uint8), (0, 0), 4), 0.7, rng.integers(0, 256, (H * 4, W * 4, 3), dtype=np.uint8), 0.3, 0)
    sign_in = np.ones(n, int)
    for t, sg in sw: sign_in[int(round(t * fps)):] = sg
    sign_eff = sign_in.copy()
    for (t, sg), d in zip(sw, delays): f0 = int(round(t * fps)); sign_eff[f0:f0 + d] = sign_in[f0 - 1]   # the model keeps the old motion for d frames
    d = os.path.join(a.selftest_dir, "selftest_ours", "synthetic__zoom__P5"); os.makedirs(d, exist_ok=True); vid = os.path.join(d, "video.mp4")
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (W, H), "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", vid], stdin=subprocess.PIPE)
    s = 1.4; cy, cx = big.shape[0] / 2, big.shape[1] / 2
    for i in range(n):
        if i > 0: s *= 1.02 ** sign_eff[i]   # motion from frame i-1 to i follows sign_eff[i]
        hh, ww = big.shape[0] / s, big.shape[1] / s; y0, x0 = int(round(cy - hh / 2)), int(round(cx - ww / 2))
        p.stdin.write(cv2.resize(big[y0:y0 + int(hh), x0:x0 + int(ww)], (W, H), interpolation=cv2.INTER_AREA).tobytes())
    p.stdin.close(); p.wait(); r = score(vid, "selftest"); got = [x["react_frames"] for x in r["switches"]]
    ok = all(g is not None and abs(g - d) <= 1 for g, d in zip(got, delays)) and r["model"] == "ours"
    print("[reaction selftest] model %s fps %g frames %d thr %.3f px ref %.3f px; injected delays %s -> measured %s (s: %s) -> %s" % (r["model"], r["fps"], r["frames"], r["thr_px"], r["ref_px"], delays, got, [x["react_s"] for x in r["switches"]], "PASS" if ok else "FAIL"))
    print("[reaction selftest] radial around the switches:", {t: [round(v, 2) for v in r["radial"][int(t * fps) - 2:int(t * fps) + 8]] for t, _ in sw})
    return r, ok
OUT = json.load(open(a.out)) if os.path.exists(a.out) else {}
if a.selftest:
    r, ok = selftest(); OUT["selftest"] = {k: v for k, v in r.items() if k not in ("radial", "horiz")}; json.dump(OUT, open(a.out, "w"), indent=1); sys.exit(0 if ok else 1)
for spec in a.specs:
    label, pat = spec.split("=", 1); vids = sorted(glob.glob(pat)); assert vids, pat
    todo = [v for v in vids if a.redo or v not in OUT or OUT[v].get("label") != label]
    print("[reaction] %s: %d videos, %d to do" % (label, len(vids), len(todo)), flush=True)
    for v in todo:
        r = score(v, label); OUT[v] = r; json.dump(OUT, open(a.out, "w"), indent=1)
        print("[reaction] %s %s model %s fps %g: %s -> mean %s frames = %s s (valid %d/%d, found %d)" % (label, os.path.relpath(v, "/home/hk28965/specwm"), r["model"], r["fps"],
              ["%s@%gs" % (s["react_frames"], s["t_s"]) for s in r["switches"]], None if r["mean_react_frames"] is None else round(r["mean_react_frames"], 1), None if r["mean_react_s"] is None else round(r["mean_react_s"], 3), r["n_valid"], r["n_switches"], r["n_found"]), flush=True)
for label in sorted({r["label"] for r in OUT.values() if isinstance(r, dict) and "label" in r}):
    rs = [r for r in OUT.values() if isinstance(r, dict) and r.get("label") == label and r.get("mean_react_s") is not None]
    if rs: print("%-22s n=%3d mean reaction %.3f s (%.1f frames)" % (label, len(rs), np.mean([r["mean_react_s"] for r in rs]), np.mean([r["mean_react_frames"] for r in rs])))
