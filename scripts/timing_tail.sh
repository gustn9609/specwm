#!/bin/bash
# after all generation jobs: student quiet steady timing + teacher-parity (cw07_safe) streaming latency, jar/castle, seed 1 = per-task warm-up
S=/home/hk28965/specwm; O=specwm_out/timing_pass; L=$S/lingbot-world-v2/$O/logs
while ps -eo args | grep -v grep | grep -q "python -u -m specwm\.\(spec_ar_loop\|spec_early\|res_split_infer\)"; do sleep 60; done
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1!=7 && $1!=6 && $2<1000{print $1; exit}'); echo "$(date) GPU $G"
R="NO_MP4=1 SAVE_LAT=0 RESERVE_GB=85 CUDA_VISIBLE_DEVICES=$G bash $S/scripts/envrun.sh python -u -m specwm."
T=sand_filling_up_jar,sandcastle_wave_impact; SD=1,1234,4321,9999
eval "${R}spec_early --model 1p3b --mode uniform:4 --tasks $T --seeds $SD --out $O --no-video --warm 0" > $L/student.log 2>&1
eval "${R}spec_early --mode diag:learned_dense --eps 0.24 --no-ref --chunk-frac 0.7 --commit safe --arm cw07_safe_stream --tasks $T --seeds $SD --out $O --no-video --warm 0 --stream-decode 1" > $L/cw07_safe_stream.log 2>&1
cd $S; source env.sh; python scripts/wm_latency.py lingbot-world-v2/$O/latency "cw07_safe=lingbot-world-v2/$O/*/cw07_safe_stream_s*" > $L/wm_latency_tail.txt 2>&1
echo "$(date) TAIL_DONE"
