#!/bin/bash
# 생성이 끝난 StEvo 출력 폴더를 자동으로 채점 (judge 서버 :30100 필요). 완료 표식: exp/Q77_DONE(h3world), Q78_DONE(h3), Q81_DONE(pretrain), Base 는 85개 video.mp4
cd /home/hk28965/specwm; R=lingbot-world-v2/specwm_out/stevo
ev() { d=$1; [ -f $R/runs/$d/summary.json ] && return; echo "[autoeval] $d $(date)"; venv/bin/python scripts/stevo_prep_eval.py $R/$d; bash scripts/stevo_eval.sh $PWD/$R/$d $PWD/$R/runs > logs/stevo_eval_$d.out 2>&1; venv/bin/python scripts/stevo_report.py > /dev/null; }
while true; do
  [ -f exp/Q77_DONE ] && ev h3world_lingbotcam
  [ -f exp/Q78_DONE ] && ev h3_video
  [ -f exp/Q81_DONE ] && ev lingbot_v2_pretrain_360_173
  if [ $(ls $R/lingbot_base/*/video.mp4 2>/dev/null | wc -l) -ge 85 ]; then mkdir -p $R/lingbot_base_flat; for v in $R/lingbot_base/*/video.mp4; do t=$(basename $(dirname $v)); ln -sfn $PWD/$v $R/lingbot_base_flat/$t.mp4; done; ev lingbot_base_flat; fi
  [ -f $R/runs/h3world_lingbotcam/summary.json ] && [ -f $R/runs/h3_video/summary.json ] && [ -f $R/runs/lingbot_v2_pretrain_360_173/summary.json ] && [ -f $R/runs/lingbot_base_flat/summary.json ] && { echo "[autoeval] ALL DONE $(date)"; exit 0; }
  sleep 600
done
