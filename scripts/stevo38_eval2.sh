#!/bin/bash
# StEvo subset eval runner v2: usage stevo38_eval2.sh PORT arm1 arm2 ...; an arm is claimed with an atomic mkdir lock, so several runners
# (one per judge) never evaluate the same arm; waits for the judge on PORT to be up.
S=/home/hk28965/specwm; O=$S/lingbot-world-v2/specwm_out; ST=$O/stevo38; PORT=$1; shift; T38=$(cat /tmp/claude-1741758895/-home-hk28965/656355b1-6d96-59ca-8160-407d814c511c/scratchpad/t38.txt)
declare -A V=( [teacher]="uniform4_s1234/video.mp4" [student]="uniform4_1p3b_s1234/video.mp4" [sdvg07]="ir_-0.7_s1234/video.mp4" [ours832]="pc1_832_s1234/cloud/spec.mp4" [ours704]="pc1_704_s1234/cloud/spec.mp4" [mg2]="mg2_s1234/video.mp4" [wp]="wp_s1234/output.mp4" [cw07_safe]="cw07_safe_s1234/video.mp4" [sdvg22]="ir_-2.2_s1234/video.mp4" [phys5]="physapi_5_s1234/video.mp4" [phys4]="physapi_4_s1234/video.mp4" )
declare -A M=( [teacher]="uniform4_s1234/loop.json" [student]="uniform4_1p3b_s1234/loop.json" [sdvg07]="ir_-0.7_s1234/loop.json" [ours832]="pc1_832_s1234/loop.json" [ours704]="pc1_704_s1234/loop.json" [mg2]="mg2_s1234/timing.json" [wp]="wp_s1234/timing.json" [cw07_safe]="cw07_safe_s1234/loop.json" [sdvg22]="ir_-2.2_s1234/loop.json" [phys5]="physapi_5_s1234/loop.json" [phys4]="physapi_4_s1234/loop.json" )
until curl -s 127.0.0.1:$PORT/v1/models | grep -q qwen; do sleep 20; done; echo "$(date) judge on $PORT up"
for arm in "$@"; do
  mkdir $ST/eval_lock_$arm 2>/dev/null || { echo "skip $arm (claimed)"; continue; }
  until n=0; for t in ${T38//,/ }; do [ -f $ST/$t/${M[$arm]} ] && n=$((n+1)); done; [ $n -ge 38 ]; do sleep 60; done
  D=$ST/eval_in/lingbot_b38_$arm; mkdir -p $D; python3 - "$D" "$arm" "${V[$arm]}" "$T38" "$ST" <<'PY'
import sys, json, os
D, arm, rel, T38, ST = sys.argv[1:6]; m = {}
for t in T38.split(","):
    src = os.path.join(ST, t, rel); dst = os.path.join(D, t + ".mp4")
    if not os.path.exists(dst): os.symlink(src, dst)
    m[t] = t + ".mp4"
json.dump(m, open(os.path.join(D, "lingbot_b38_%s.json" % arm), "w"), indent=1)
PY
  echo "$(date) eval $arm on $PORT"; JUDGE_PORT=$PORT bash $S/scripts/stevo_eval.sh $D $ST/eval_runs > $ST/logs/eval_$arm.log 2>&1; echo "$(date) done $arm"
done
echo "$(date) RUNNER_$PORT_DONE"
