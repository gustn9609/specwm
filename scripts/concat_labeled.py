"""Side-by-side comparison video with a caption bar over each panel (PIL; ffmpeg here has no drawtext).
usage: concat_labeled.py OUT.mp4 --fps 16 --height 288 "label1=path1.mp4" "label2=path2.mp4" ...  (frames cut to the shortest clip)"""
import sys, argparse, numpy as np, imageio
from PIL import Image, ImageDraw, ImageFont
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("items", nargs="+"); ap.add_argument("--fps", type=float, default=16); ap.add_argument("--height", type=int, default=288); ap.add_argument("--max-frames", type=int, default=0); a = ap.parse_args()
try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
except Exception: font = ImageFont.load_default()
clips = []
for it in a.items:
    lab, path = it.split("=", 1); fr = [np.asarray(f) for f in imageio.get_reader(path)]; clips.append((lab, fr))
n = min(len(fr) for _, fr in clips)
if a.max_frames: n = min(n, a.max_frames)
H = a.height; bar = 30; panels_w = []
for lab, fr in clips: h, w = fr[0].shape[:2]; panels_w.append(int(round(w * H / h / 2)) * 2)
W = sum(panels_w) + 4 * (len(clips) - 1)
wr = imageio.get_writer(a.out, fps=a.fps, quality=8, macro_block_size=2)
for t in range(n):
    canvas = Image.new("RGB", (W, H + bar), (18, 18, 18)); d = ImageDraw.Draw(canvas); x = 0
    for (lab, fr), pw in zip(clips, panels_w):
        canvas.paste(Image.fromarray(fr[t]).resize((pw, H), Image.BICUBIC), (x, bar)); d.text((x + 8, 6), lab, fill=(240, 240, 240), font=font); x += pw + 4
    wr.append_data(np.asarray(canvas))
wr.close(); print("[concat] %s %d frames, %dx%d" % (a.out, n, W, H + bar))
