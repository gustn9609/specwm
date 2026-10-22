"""[10/19] One side-by-side comparison video per (backbone scene set, scene, program) from the frozen baseline runs, plus one reel per set.
Every tile: same start image + camera program, resampled to 16 fps, first 10 s, synchronized. Labels/numbers are drawn on a static background (PIL)
because the static ffmpeg has no drawtext. usage: tools/vbench_env/bin/python paper/1019/make_grids.py [CLIPS_CSV]"""
import csv, json, os, sys, glob, subprocess
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw, ImageFont
B = "/home/hk28965/specwm/baselines"; OUT = os.path.dirname(os.path.abspath(__file__)); FF = "/home/hk28965/specwm/tools/ffmpeg-7.0.2-amd64-static/ffmpeg"
CSV = sys.argv[1] if len(sys.argv) > 1 else ""
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"; FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SETS = [("mg2", "mg2", "Matrix-Game 2.0"), ("wp", "waypoint", "Waypoint-1.5 1B"), ("abot", "abot", "ABot-World-0 5B"), ("astra", "astra", "Astra")]
ARMS = [  # (suffix or None = baseline, label, colour); hybrid cloud timing comes from the device stream
    (None, "Baseline: {B}", (181, 84, 27)),
    ("sf1p3b_pub", "Speculative Forcing, its setting (1.3B, published thr.)", (122, 75, 184)),
    ("sf1p3b_cal", "Speculative Forcing, its setting (1.3B, recalibrated)", (122, 75, 184)),
    ("sf24", "Speculative Forcing on LingBot 14B", (122, 75, 184)),
    ("teacher", "LingBot-World 14B 4-step (base)", (90, 98, 112)),
    ("rt2", "LingBot-World 14B 2-step (naive few-step)", (90, 98, 112)),
    ("adapt25", "Ours w/o cloud (ablation)", (31, 95, 209)),
    ("adapt25c1hyb_dev", "OURS (device screen)", (15, 122, 92)),
    ("adapt25c1hyb_cloud", "Ours, cloud-corrected stream (1-5 s behind)", (15, 122, 92))]
PROG = {"P1": "P1 forward", "P2": "P2 turn right 18 deg/s (180 deg pan)", "P2p": "P2p turn right 16 deg/s (160 deg pan)", "P3": "P3 forward + turn right 12 deg/s",
        "P4": "P4 S-curve, yaw +/-18 deg/s (2 reversals)", "P4p": "P4p S-curve, yaw +/-16 deg/s", "P5": "P5 stop-and-go: forward, stop, back up, forward"}
TW, LH, COLS = 480, 46, 4
M = {}
if CSV:
    for r in csv.DictReader(open(CSV)): M[r["video"]] = r
def num(r, k):
    try: return float(r[k])
    except Exception: return None
def tile_text(arm, suf, vid):
    tdir = os.path.dirname(vid).replace("_cloud_", "_dev_") if suf and suf.endswith("_cloud") else os.path.dirname(vid)
    try: T = json.load(open(tdir + "/timing.json"))
    except Exception: T = {}
    g = T.get("gen_s"); r = M.get(vid, {}); q, d = num(r, "judge_q"), num(r, "judge_drift")
    s = ["10-s clip generated in %.1f s" % g if g else "gen time n/a"]
    s.append("judge Q %.0f / drift %.0f" % (q, d) if q is not None and d is not None else "not scored yet")
    return "  |  ".join(s)
def background(path, tiles, th, info):
    rows = -(-(len(tiles) + 1) // COLS); W, H = TW * COLS, rows * (th + LH)
    im = Image.new("RGB", (W, H), (16, 18, 22)); dr = ImageDraw.Draw(im)
    fb, fs, fi = ImageFont.truetype(FONTB, 15), ImageFont.truetype(FONT, 13), ImageFont.truetype(FONT, 16)
    for i, (lab, sub, col) in enumerate(tiles):
        x, y = (i % COLS) * TW, (i // COLS) * (th + LH)
        dr.rectangle([x, y, x + TW - 1, y + 4], fill=col); dr.text((x + 8, y + 7), lab, font=fb, fill=(240, 242, 245)); dr.text((x + 8, y + 27), sub, font=fs, fill=(170, 176, 186))
    i = len(tiles); x, y = (i % COLS) * TW, (i // COLS) * (th + LH)
    for k, line in enumerate(info): dr.text((x + 12, y + 12 + 24 * k), line, font=fb if k == 0 else fi, fill=(240, 242, 245) if k == 0 else (190, 196, 206))
    im.save(path); return W, H
def build(job):
    out, bg, vids, th = job
    if os.path.exists(out): return out
    args = [FF, "-v", "error", "-y", "-loop", "1", "-framerate", "16", "-i", bg]
    for v in vids: args += ["-i", v]
    fl, last = [], "0:v"
    for i, v in enumerate(vids):
        fl.append("[%d:v]fps=16,scale=%d:%d,setsar=1,tpad=stop_mode=clone:stop_duration=2,trim=duration=10,setpts=PTS-STARTPTS[t%d]" % (i + 1, TW, th, i))
    for i, v in enumerate(vids):
        x, y = (i % COLS) * TW, (i // COLS) * (th + LH) + LH
        fl.append("[%s][t%d]overlay=%d:%d:shortest=0:eof_action=repeat[o%d]" % (last, i, x, y, i)); last = "o%d" % i
    args += ["-filter_complex", ";".join(fl), "-map", "[%s]" % last, "-t", "10", "-r", "16", "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", "-threads", "3", out]
    subprocess.run(["nice", "-n", "15"] + args, check=True); return out
jobs, reels = [], {}
for sid, barm, blab in SETS:
    d = os.path.join(OUT, sid); os.makedirs(d + "/.bg", exist_ok=True)
    clips = sorted(c for c in os.listdir(os.path.join(B, "runs", barm)) if c.startswith(sid + "__") and c.split("__")[-1] in PROG)
    order = {p: i for i, p in enumerate(PROG)}; clips.sort(key=lambda c: (c.split("__")[1], order[c.split("__")[-1]]))
    for c in clips:
        _, scene, P = c.split("__"); tiles, vids = [], []
        for suf, lab, col in ARMS:
            arm = barm if suf is None else "ours_%s_%s" % (suf, sid); v = os.path.join(B, "runs", arm, c, "video.mp4")
            if not os.path.exists(v): continue
            tiles.append((lab.format(B=blab), tile_text(arm, suf, v), col)); vids.append(v)
        T = json.load(open(os.path.join(B, "runs", barm, c, "timing.json"))); res = T.get("resolution_out"); res = "x".join(map(str, res)) if isinstance(res, list) else res
        w, h = map(int, res.split("x")); th = int(round(TW * h / w / 2)) * 2
        info = ["%s scenes" % blab, "scene: %s" % scene.replace("_", " "), PROG[P], "resolution %s (baseline native; all arms)" % res,
                "same start image + camera program", "all tiles 16 fps, first 10 s, in sync", "judge = Qwen3-VL-32B (Q 1-10 up, drift 0-100 down)"]
        bg = os.path.join(d, ".bg", c + ".png"); background(bg, tiles, th, info)
        out = os.path.join(d, "%s__%s.mp4" % (scene, P)); jobs.append((out, bg, vids, th)); reels.setdefault(sid, []).append(out)
if os.environ.get("ONLY"): jobs = [j for j in jobs if j[0].endswith(os.environ["ONLY"])]; reels = {}
print("jobs", len(jobs), flush=True)
with ThreadPoolExecutor(6) as ex:
    for i, o in enumerate(ex.map(build, jobs)):
        if i % 20 == 0: print("done", i + 1, "/", len(jobs), o, flush=True)
for sid, outs in reels.items():   # one reel per set: all its grids back to back (same encoder settings -> stream copy)
    lst = os.path.join(OUT, sid, ".bg", "concat.txt"); open(lst, "w").write("".join("file '%s'\n" % o for o in outs))
    subprocess.run([FF, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags", "+faststart", os.path.join(OUT, "%s_ALL_%dclips.mp4" % (sid, len(outs)))], check=True)
    print("reel", sid, len(outs), flush=True)
