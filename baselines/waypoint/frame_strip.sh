#!/bin/bash
# Contact sheets of a clip for eyeballing:  frame_strip.sh VIDEO.mp4 OUT_PREFIX
#   OUT_PREFIX_strip6.png  = the PROTOCOL smoke strip (every 32nd frame, first 6 tiles = frames 0..160, 300 px wide each)
#   OUT_PREFIX_sec.png     = one tile per second (every 60th frame at 60 fps), 10 tiles in one row
#   OUT_PREFIX_all32.png   = every 32nd frame, all 19 tiles of a 600-frame clip (10x2 grid)
set -e; F=/home/hk28965/specwm/tools/ffmpeg-7.0.2-amd64-static/ffmpeg; [ -x $F ] || F=ffmpeg
V=$1; O=$2
$F -y -loglevel error -i "$V" -vf "select='not(mod(n\,32))',scale=300:-1,tile=6x1" -frames:v 1 "${O}_strip6.png"
$F -y -loglevel error -i "$V" -vf "select='not(mod(n\,60))',scale=240:-1,tile=10x1" -frames:v 1 "${O}_sec.png"
$F -y -loglevel error -i "$V" -vf "select='not(mod(n\,32))',scale=200:-1,tile=10x2" -frames:v 1 "${O}_all32.png"
ls -la "${O}"_*.png
