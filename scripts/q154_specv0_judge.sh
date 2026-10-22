#!/bin/bash
# spec v0: wait for both task jobs, then StEvo-judge every arm/seed (incl. Base seeds) on the two failure tasks + DINO eval. Judge on GPU ${GPU:-3}.
cd /home/hk28965/specwm; R=lingbot-world-v2/specwm_out; V=$R/spec_v0
while [ ! -f exp/Q152_sand_filling_up_jar_DONE ] || [ ! -f exp/Q152_sandcastle_wave_impact_DONE ]; do sleep 300; done
ARMS=""; for d in $V/*/*_s*; do [ -f $d/video.mp4 ] || continue; a=$(basename $d); t=$(basename $(dirname $d)); o=$R/stevo/lingbot_specv0_$a; mkdir -p $o; ln -sfn $PWD/$d/video.mp4 $o/$t.mp4; ARMS="$ARMS $a"; done
GPU=${GPU:-3} setsid nohup bash scripts/judge_serve.sh > logs/judge_serve.out 2>&1 < /dev/null & for i in $(seq 1 120); do curl -s --max-time 3 http://127.0.0.1:30100/health >/dev/null && break; sleep 10; done; sleep 20
for a in $(echo $ARMS | tr ' ' '\n' | sort -u); do d=lingbot_specv0_$a; venv/bin/python scripts/stevo_prep_eval.py $R/stevo/$d; bash scripts/stevo_eval.sh $PWD/$R/stevo/$d $PWD/$R/stevo/runs > logs/stevo_eval_$d.out 2>&1; echo "[judged $a] $(date)"; done
pkill -f "sglang.launch_serve[r]"
CUDA_VISIBLE_DEVICES=${GPU:-3} venv/bin/python scripts/spec_v0_eval.py > logs/spec_v0_eval.out 2>&1
touch exp/Q154_DONE; echo "[Q154_DONE] $(date)"
