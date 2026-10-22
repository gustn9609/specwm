#!/bin/bash
# Contact strips of an Astra clip (832x480, 20 fps, 201 frames) for eyeballing:  frame_strip.sh VIDEO.mp4 OUT_PREFIX
#   OUT_PREFIX_sec.png   = one tile per second (every 20th frame: f0, f20, ..., f200), 11 tiles in one row (300 px wide each)
#   OUT_PREFIX_end.png   = the last 2 s every 4th frame (f160 ... f200), 11 tiles: shows whether the trimmed tail (block 7, frames 193-200) is clean
#   OUT_PREFIX_q.png     = every 8th frame, 26 tiles in a 13x2 grid (200 px wide each)
set -e; F=/home/hk28965/.local/bin/ffmpeg; [ -x $F ] || F=ffmpeg
V=$1; O=$2
$F -y -loglevel error -i "$V" -vf "select='not(mod(n\,20))',scale=300:-1,tile=11x1" -frames:v 1 "${O}_sec.png"
$F -y -loglevel error -i "$V" -vf "select='gte(n\,160)*not(mod(n\,4))',scale=300:-1,tile=11x1" -frames:v 1 "${O}_end.png"
$F -y -loglevel error -i "$V" -vf "select='not(mod(n\,8))',scale=200:-1,tile=13x2" -frames:v 1 "${O}_q.png"
ls -la "${O}"_*.png
