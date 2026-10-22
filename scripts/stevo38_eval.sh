#!/bin/bash
# StEvo-Bench subset (38 camera-WM tasks, seed 1234, 253-frame camera look-away path): for each arm, wait until its 38 clips are finished,
# build the StEvo outputs folder (lingbot_b38_<arm>/<task>.mp4 + output map; the "lingbot" prefix selects the camera-controlled protocol)
# and run the StEvo evaluation (control, physics, state; 3-vote majority; local Qwen3-VL-32B judge on :30100).
S=/home/hk28965/specwm; O=$S/lingbot-world-v2/specwm_out; ST=$O/stevo38; T38=$(cat /tmp/claude-1741758895/-home-hk28965/656355b1-6d96-59ca-8160-407d814c511c/scratchpad/t38.txt)
declare -A V=( [teacher]="uniform4_s1234/video.mp4" [student]="uniform4_1p3b_s1234/video.mp4" [sdvg07]="ir_-0.7_s1234/video.mp4" [sdvg22]="ir_-2.2_s1234/video.mp4"
               [ours832]="pc1_832_s1234/cloud/spec.mp4" [ours704]="pc1_704_s1234/cloud/spec.mp4" [mg2]="mg2_s1234/video.mp4" [wp]="wp_s1234/output.mp4" )
declare -A M=( [teacher]="uniform4_s1234/loop.json" [student]="uniform4_1p3b_s1234/loop.json" [sdvg07]="ir_-0.7_s1234/loop.json" [sdvg22]="ir_-2.2_s1234/loop.json"
               [ours832]="pc1_832_s1234/loop.json" [ours704]="pc1_704_s1234/loop.json" [mg2]="mg2_s1234/timing.json" [wp]="wp_s1234/timing.json" )
DONE=""
while true; do
  for arm in teacher ours704 ours832 mg2 student sdvg07 wp; do
    [[ " $DONE " == *" $arm "* ]] && continue
    n=0; for t in ${T38//,/ }; do [ -f $ST/$t/${M[$arm]} ] && n=$((n+1)); done
    [ $n -lt 38 ] && continue
    D=$ST/eval_in/lingbot_b38_$arm; mkdir -p $D; python3 - "$D" "$arm" "${V[$arm]}" "$T38" "$ST" <<'PY'
import sys, json, os
D, arm, rel, T38, ST = sys.argv[1:6]; m = {}
for t in T38.split(","):
    src = os.path.join(ST, t, rel); dst = os.path.join(D, t + ".mp4")
    if not os.path.exists(dst): os.symlink(src, dst)
    m[t] = t + ".mp4"
json.dump(m, open(os.path.join(D, "lingbot_b38_%s.json" % arm), "w"), indent=1)
PY
    echo "$(date) eval $arm"; bash $S/scripts/stevo_eval.sh $D $ST/eval_runs > $ST/logs/eval_$arm.log 2>&1; echo "$(date) done $arm"; DONE="$DONE $arm"
  done
  [ $(echo $DONE | wc -w) -ge 7 ] && break; sleep 120
done
echo "$(date) STEVO_EVAL_DONE"
