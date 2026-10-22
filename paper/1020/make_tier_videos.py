"""[10/20] Device-tier comparison videos at 832x480: original (FP8 weights, bf16 KV) | 32 GB class (FP8 weights, FP8 KV) | 24 GB class (GPTQ-4 weights,
FP8 KV). Clips = the 10/13 fit24 harness runs (lingbot-world-v2/specwm_out/fit24/<cfg>/<task>/spec.mp4: 14B 2-step, window 18, same seed / prompt /
camera per task), labels = resolution + generation peak memory (torch reserved) only; the harness's fp32 full decode adds ~7 GiB on top (not shown; the device uses the bf16 2x2 tiled decode, +1.5 GiB).
Output: paper/1020/videos/tiers832_<task>.mp4 and tiers832_ALL.mp4.   usage: python3 -I make_tier_videos.py   (CPU only, ffmpeg + PIL from tools/vbench_env)"""
import os, json, glob, subprocess, sys, numpy as np
sys.path.insert(0, "/home/hk28965/specwm/tools/vbench_env/lib/python3.10/site-packages"); sys.path.insert(0, "/home/hk28965/specwm/tools/vbench_env/lib/python3.11/site-packages")
from PIL import Image, ImageDraw, ImageFont
S = "/home/hk28965/specwm"; F24 = S + "/lingbot-world-v2/specwm_out/fit24"; OUT = S + "/paper/1020/videos"; FF = S + "/tools/ffmpeg-7.0.2-amd64-static/ffmpeg"; FP = os.path.dirname(FF) + "/ffprobe"
CFGS = [("a832_fp8_bf16", "original: 14B FP8 weights, bf16 KV"), ("a832_fp8_fp8kv", "32 GB class: FP8 weights, FP8 KV"), ("q43_a832_gptq4_fp8kv", "24 GB class: GPTQ-4 weights, FP8 KV")]
TW, TH, LH = 416, 240, 50; FPS = 16
try: FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15); FONT2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
except Exception: FONT = FONT2 = ImageFont.load_default()
def peaks(cfg):
    r = json.load(open("%s/%s/fit.json" % (F24, cfg))); return r["peak_gen_reserved_GiB"], r.get("peak_reserved_GiB")
def probe(p):
    o = subprocess.run([FP, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames,r_frame_rate", "-of", "json", p], capture_output=True, text=True).stdout
    s = json.loads(o)["streams"][0]; return int(s["width"]), int(s["height"]), int(s["nb_frames"])
def read(p, n):
    w, h, nf = probe(p); raw = subprocess.run([FF, "-v", "error", "-i", p, "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3), (w, h)
tasks = sorted(set.intersection(*[set(os.path.basename(os.path.dirname(p)) for p in glob.glob("%s/%s/*/spec.mp4" % (F24, c))) for c, _ in CFGS]))
print("common tasks:", tasks)
labels = []
for c, lab in CFGS:
    g, t = peaks(c); labels.append((lab, "832x480   peak %.0f GiB" % g))
made = []
for task in tasks:
    n = min(probe("%s/%s/%s/spec.mp4" % (F24, c, task))[2] for c, _ in CFGS); vids = [read("%s/%s/%s/spec.mp4" % (F24, c, task), n)[0] for c, _ in CFGS]
    n = min(len(v) for v in vids); out = "%s/tiers832_%s.mp4" % (OUT, task)
    p = subprocess.Popen([FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (TW * 3, LH + TH), "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    for f in range(n):
        canvas = Image.new("RGB", (TW * 3, LH + TH), (18, 18, 18)); d = ImageDraw.Draw(canvas)
        for i, (v, (l1, l2)) in enumerate(zip(vids, labels)):
            canvas.paste(Image.fromarray(v[f]).resize((TW, TH), Image.BILINEAR), (i * TW, LH)); d.text((i * TW + 6, 5), l1, font=FONT, fill=(240, 240, 240)); d.text((i * TW + 6, 27), l2, font=FONT2, fill=(255, 200, 90))
        d.text((TW * 3 - 6 - d.textlength(task, font=FONT2), LH - 18), task, font=FONT2, fill=(140, 140, 140))
        p.stdin.write(np.asarray(canvas).tobytes())
    p.stdin.close(); p.wait(); made.append(out); print("wrote", out, n)
lst = OUT + "/.concat_tiers.txt"; open(lst, "w").write("".join("file '%s'\n" % m for m in made))
subprocess.run([FF, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", OUT + "/tiers832_ALL.mp4"], check=True); print("wrote ALL", len(made))
