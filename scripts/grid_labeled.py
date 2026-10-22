"""Grid comparison video with a caption bar over each panel, time-aligned across clips of different frame rates (PIL; ffmpeg here has no drawtext).
usage: grid_labeled.py OUT.mp4 --fps 16 --height 288 --cols 4 [--seconds 9.5] [--title "..."] "label=path.mp4[@src_fps]" ...
Each clip is sampled by wall-clock time: output frame t uses source frame floor(t / fps * src_fps) (src_fps defaults to the file's rate);
the output length is the shortest clip in seconds (or --seconds). Panels are scaled to a common height; rows are filled left to right."""
import argparse, numpy as np, imageio
from PIL import Image, ImageDraw, ImageFont
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("items", nargs="+"); ap.add_argument("--fps", type=float, default=16); ap.add_argument("--height", type=int, default=288)
ap.add_argument("--cols", type=int, default=4); ap.add_argument("--seconds", type=float, default=0); ap.add_argument("--title", default=""); a = ap.parse_args()
try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18); tfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
except Exception: font = tfont = ImageFont.load_default()
clips = []
for it in a.items:
    lab, path = it.split("=", 1); sfps = None
    if "@" in path: path, sfps = path.rsplit("@", 1); sfps = float(sfps)
    if path.lower().endswith((".png", ".jpg", ".jpeg")): fr = [np.asarray(Image.open(path).convert("RGB"))]; sfps = 1.0   # static panel (e.g. the start image)
    else: rd = imageio.get_reader(path); meta = rd.get_meta_data(); fr = [np.asarray(f) for f in rd]; sfps = sfps or float(meta.get("fps", a.fps))
    clips.append((lab, fr, sfps))
dur = min(len(fr) / sfps for _, fr, sfps in clips if len(fr) > 1)
if a.seconds: dur = min(dur, a.seconds)
n = int(dur * a.fps); H = a.height; bar = 30; gap = 4; tbar = 36 if a.title else 0
pw = [int(round(fr[0].shape[1] * H / fr[0].shape[0] / 2)) * 2 for _, fr, _ in clips]
rows = [list(range(i, min(i + a.cols, len(clips)))) for i in range(0, len(clips), a.cols)]
W = max(sum(pw[j] for j in r) + gap * (len(r) - 1) for r in rows); Ht = tbar + len(rows) * (H + bar) + gap * (len(rows) - 1)
wr = imageio.get_writer(a.out, fps=a.fps, quality=8, macro_block_size=2)
for t in range(n):
    canvas = Image.new("RGB", (W, Ht), (18, 18, 18)); d = ImageDraw.Draw(canvas)
    if a.title: d.text((8, 7), a.title, fill=(240, 240, 240), font=tfont)
    y = tbar
    for r in rows:
        x = 0
        for j in r:
            lab, fr, sfps = clips[j]; k = min(len(fr) - 1, int(t / a.fps * sfps))
            canvas.paste(Image.fromarray(fr[k]).resize((pw[j], H), Image.BICUBIC), (x, y + bar)); d.text((x + 8, y + 6), lab, fill=(240, 240, 240), font=font); x += pw[j] + gap
        y += H + bar + gap
    wr.append_data(np.asarray(canvas))
wr.close(); print("[grid] %s %d frames (%.1f s), %dx%d" % (a.out, n, dur, W, Ht))
