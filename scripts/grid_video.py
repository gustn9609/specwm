#!/usr/bin/env python3
"""[10/22] 2x2 labeled comparison video (PIL labels, frames piped to ffmpeg): grid_video.py OUT.mp4 TITLE 'label0' v0.mp4 'label1' v1.mp4 'label2' v2.mp4 'label3' v3.mp4 [--montage OUT.png --at 0,15,30,45,59]"""
import sys, subprocess, argparse, numpy as np, decord
from PIL import Image, ImageDraw, ImageFont
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("title"); ap.add_argument("items", nargs=8); ap.add_argument("--montage", default=""); ap.add_argument("--at", default="0,15,30,45,59")
a = ap.parse_args(); labels = a.items[0::2]; vids = a.items[1::2]
F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18); Fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
vr = [decord.VideoReader(v, num_threads=2) for v in vids]; n = min(len(x) for x in vr); fps = float(vr[0].get_avg_fps()); H, W = vr[0][0].shape[:2]
def grid(i):
    im = Image.new("RGB", (2 * W, 2 * H + 30), "black")
    for k, r in enumerate(vr):
        f = Image.fromarray(r[min(i, len(r) - 1)].asnumpy()); d = ImageDraw.Draw(f); tw = d.textlength(labels[k], font=F)
        d.rectangle([4, 4, 12 + tw, 30], fill=(0, 0, 0, 160)); d.text((8, 6), labels[k], font=F, fill="white"); im.paste(f, ((k % 2) * W, (k // 2) * H))
    d = ImageDraw.Draw(im); t = "%s   t = %4.1f s" % (a.title, i / fps); d.text(((2 * W - d.textlength(t, font=Fs)) / 2, 2 * H + 6), t, font=Fs, fill="white"); return im
p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (2 * W, 2 * H + 30), "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
for i in range(n): p.stdin.write(np.asarray(grid(i)).tobytes())
p.stdin.close(); p.wait(); print("wrote", a.out, n, "frames")
if a.montage:
    ts = [float(x) for x in a.at.split(",")]; cols = [grid(min(n - 1, int(round(t * fps)))) for t in ts]
    m = Image.new("RGB", (cols[0].width * len(cols), cols[0].height), "black")
    for k, c in enumerate(cols): m.paste(c, (k * c.width, 0))
    m.save(a.montage); print("wrote", a.montage)
