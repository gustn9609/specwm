#!/bin/bash
# ABot-World-0-5B-LF on our 12 prompts (task image + camera_WM caption + camera path → 8 keys per block), 13 blocks, FP8 per-tensor + taew2_2,
# at its native 1280x704 and at our 704x384. GPU 2 (after the resolution smoke chain). Then judge into vlm_a704_43.json (labels a704_43_abot1280 / abot704).
S=/home/hk28965/specwm; O=$S/lingbot-world-v2/specwm_out/abot; R=$S/scripts/envrun.sh
until grep -q ABOT_SMOKE_CHAIN_DONE $O/logs/smoke_chain.out 2>/dev/null; do sleep 60; done
T="balloon_block_cut billiard_break bowling_inertia domino_falling easy_dragon_night easy_greatwall easy_lake easy_urban_rain inflating_balloon newtons_cradle pouring_water_into_cup traffic_light_green"
for t in $T; do
  [ -f $O/keys/$t.json ] || $R python $S/tools/abot/poses_to_keys.py $t $O/keys/$t.json > /dev/null 2>&1
  IMG=$($R python -c "import json;T={x['id']:x for x in json.load(open('$S/data/stevo/tasks_camera.json'))};print(T['$t']['image'])" 2>/dev/null | tail -1)
  PR=$($R python -c "import json;T={x['id']:x for x in json.load(open('$S/data/stevo/tasks_camera.json'))};print(T['$t']['camera_WM'].replace(chr(10),' '))" 2>/dev/null | tail -1)
  for r in 1280x704 704x384; do W=${r%x*}; H=${r#*x}; [ -f $O/a$W/$t/video.mp4 ] && continue
    CUDA_VISIBLE_DEVICES=2 $S/tools/abot/run.sh python bench_ours.py --image "$IMG" --prompt "$PR" --actions "$(cat $O/keys/$t.json)" --out $O/a$W/$t --blocks 13 --width $W --height $H --quant fp8-per-tensor > $O/logs/a${W}_$t.log 2>&1
    grep "ABOT_BENCH_DONE\|Error" $O/logs/a${W}_$t.log | tail -1 | cut -c1-200
  done
done
JUDGE_PORT=30100 flock /tmp/vlm_a704_43.lock $R python $S/scripts/vlm_clip_score.py $S/lingbot-world-v2/specwm_out/twomach/vlm_a704_43.json "a704_43_abot1280=$O/a1280/*/video.mp4" "a704_43_abot704=$O/a704/*/video.mp4"
echo ABOT12_DONE
