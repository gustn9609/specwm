#!/bin/bash
# 재접지 arms 생성이 끝나면 judge 를 GPU 5 에 올려 StEvo 채점: fast_alone(=Fast 재현), resync_k6(ideal 재접지), kvrw_laundered(방법)
cd /home/hk28965/specwm; R=lingbot-world-v2/specwm_out/stevo
while true; do n=$(ls $R/reground/*/W18_sink6/kvrw_laundered_s1234/video.mp4 2>/dev/null | wc -l); [ $n -ge 85 ] && break; sleep 900; done
for arm in fast_alone_s1234 resync_k6_s1234 kvrw_laundered_s1234; do d=$R/lingbot_reground_${arm%_s1234}; mkdir -p $d; for v in $R/reground/*/W18_sink6/$arm/video.mp4; do t=$(basename $(dirname $(dirname $(dirname $v)))); ln -sfn $PWD/$v $d/$t.mp4; done; done
GPU=5 setsid nohup bash scripts/judge_serve.sh > logs/judge_serve.out 2>&1 < /dev/null & for i in $(seq 1 120); do curl -s --max-time 3 http://127.0.0.1:30100/health >/dev/null && break; sleep 10; done; sleep 20
for arm in fast_alone resync_k6 kvrw_laundered; do d=lingbot_reground_$arm; venv/bin/python scripts/stevo_prep_eval.py $R/$d; bash scripts/stevo_eval.sh $PWD/$R/$d $PWD/$R/runs > logs/stevo_eval_$d.out 2>&1; done
[ -d $R/longlive2_5b_video ] && { venv/bin/python scripts/stevo_prep_eval.py $R/longlive2_5b_video; bash scripts/stevo_eval.sh $PWD/$R/longlive2_5b_video $PWD/$R/runs > logs/stevo_eval_longlive2_5b_video.out 2>&1; }
pkill -f "sglang.launch_serve[r]"; echo "[Q105_DONE] $(date)"
