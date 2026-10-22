"""[baselines/sf1p3b] contact sheet of two clips for a visual check: rows = videos, columns = the requested frame ids (downscaled).
usage: python contact_sheet.py OUT.png VIDEO_A VIDEO_B [--frames 0,24,48,72,96,120,152] [--width 320]"""
import sys, argparse, numpy as np, imageio.v2 as imageio
from PIL import Image, ImageDraw
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("videos", nargs="+"); ap.add_argument("--frames", default="0,24,48,72,96,120,152"); ap.add_argument("--width", type=int, default=320); a = ap.parse_args()
ids = [int(x) for x in a.frames.split(",")]; rows = []
for v in a.videos:
    fr = imageio.mimread(v, memtest=False); n = len(fr); tiles = []
    for i in ids:
        im = Image.fromarray(fr[min(i, n - 1)]); im = im.resize((a.width, int(a.width * im.height / im.width))); ImageDraw.Draw(im).text((4, 4), "%s f%d/%d" % (v.split("/")[-3][:28], min(i, n - 1), n), fill=(255, 255, 0)); tiles.append(im)
    rows.append(tiles)
W = a.width * len(ids); H = sum(r[0].height for r in rows); sheet = Image.new("RGB", (W, H)); y = 0
for r in rows:
    for j, im in enumerate(r): sheet.paste(im, (j * a.width, y))
    y += r[0].height
sheet.save(a.out); print(a.out, sheet.size)
