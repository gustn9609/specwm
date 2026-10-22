#!/bin/bash
# 생성 완료 시 전체 재채점: base 85 / h3 221 / pretrain 85 (judge :30100). 완료된 것은 runs/<m>/.final 표식.
cd /home/hk28965/specwm; R=lingbot-world-v2/specwm_out/stevo
ev() { d=$1; n=$2; [ -f $R/runs/$d/.final ] && return; c=$(ls $R/$d/*.mp4 2>/dev/null | wc -l); [ $c -ge $n ] || return; echo "[autoeval2] $d $c $(date)"; rm -f $R/runs/$d/summary.json; venv/bin/python scripts/stevo_prep_eval.py $R/$d; bash scripts/stevo_eval.sh $PWD/$R/$d $PWD/$R/runs > logs/stevo_eval_${d}_final.out 2>&1 && touch $R/runs/$d/.final; venv/bin/python scripts/stevo_report.py > /dev/null; }
while true; do
  mkdir -p $R/lingbot_base_flat; for v in $R/lingbot_base/*/video.mp4; do t=$(basename $(dirname $v)); ln -sfn $PWD/$v $R/lingbot_base_flat/$t.mp4; done
  ev lingbot_base_flat 85; ev h3_video 221; ev lingbot_v2_pretrain_360_173 85
  [ -f $R/runs/lingbot_base_flat/.final ] && [ -f $R/runs/h3_video/.final ] && [ -f $R/runs/lingbot_v2_pretrain_360_173/.final ] && { echo "[autoeval2] ALL DONE $(date)"; exit 0; }
  sleep 900
done
