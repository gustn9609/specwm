#!/bin/bash
# All-baseline comparison videos (10/16): per prompt a 2-row grid — row 1 = start image + LingBot-World 14B (teacher 832, ours realtime 704, ours 1280),
# row 2 = edge-only baselines (Waypoint-1.5-1B, Matrix-Game 2.0, Matrix-Game 3.5, ABot-World-0). Time-aligned (ABot 12 fps resampled). Output
# paper/meeting_1012/videos/baselines/<task>.mp4 + all10.mp4 (prompts back to back). usage: build_baseline_videos.sh [task ...]
S=/home/hk28965/specwm; O=$S/lingbot-world-v2/specwm_out; V=$S/paper/meeting_1012/videos/baselines; mkdir -p $V; R=$S/scripts/envrun.sh
T=${@:-"easy_lake easy_greatwall easy_stonehenge_sunset easy_urban_rain easy_dragon_night pouring_water_into_cup domino_falling hourglass_accumulation newtons_cradle billiard_break"}
for t in $T; do
  IMG=$($R python -c "import json;T={x['id']:x for x in json.load(open('$S/data/stevo/tasks_camera.json'))};print(T['$t']['image'])" 2>/dev/null | tail -1)
  for f in $O/twomach/a832/teacher/$t/spec.mp4 $O/twomach/a704_43/rt2/$t/spec.mp4 $O/twomach/a1280/rt2/$t/spec.mp4 $O/waypoint/cmp10/$t/output.mp4 $O/mg2/cmp10/$t/video.mp4 $O/mg35/cmp10/$t/result.mp4 $O/abot/a1280/$t/video.mp4; do
    [ -f $f ] || { echo "[skip $t] missing $f"; continue 2; }; done
  $R python $S/scripts/grid_labeled.py $V/$t.mp4 --fps 16 --height 300 --cols 4 --title "$t  (same start image, same camera path; 1 RTX PRO 6000 each; fps = generated frames/s shown)" \
    "start image=$IMG" \
    "LingBot 14B teacher 4-step 832x480 | 4.1 fps=$O/twomach/a832/teacher/$t/spec.mp4" \
    "ours: 14B 2-step 704x384 (realtime mode) | 15.7 fps=$O/twomach/a704_43/rt2/$t/spec.mp4" \
    "ours: 14B 2-step 1280x704 | 3.7 fps=$O/twomach/a1280/rt2/$t/spec.mp4" \
    "Waypoint-1.5-1B 1024x512 kb+mouse | 28 fps=$O/waypoint/cmp10/$t/output.mp4" \
    "Matrix-Game 2.0 1.8B 640x352 kb+mouse | 20.5 fps=$O/mg2/cmp10/$t/video.mp4" \
    "Matrix-Game 3.5 5B 1280x704 camera+text | 0.9 fps=$O/mg35/cmp10/$t/result.mp4" \
    "ABot-World-0 5B 1280x704 kb | 13.3 fps=$O/abot/a1280/$t/video.mp4@12" 2>&1 | grep -v Warning | tail -1
done
ls $V/*.mp4 | grep -v 'all10' | grep -v '_small' | sed "s|^|file '|; s|$|'|" > $V/list.txt
ffmpeg -y -v error -f concat -safe 0 -i $V/list.txt -c copy $V/all10.mp4 && ffmpeg -y -v error -i $V/all10.mp4 -vf scale=1200:-2 -c:v libx264 -crf 30 -preset fast -pix_fmt yuv420p $V/all10_small.mp4 && echo "[all] $V/all10.mp4 $(ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames -of csv=p=0 $V/all10.mp4)"
