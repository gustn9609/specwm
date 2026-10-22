#!/bin/bash
# after the disentangling runs: smoke-test the target-KV / mc code on 4+5, then run it on both pairs
cd /home/hk28965/specwm
GPUS=4,5 SMOKE=1 TAG=smoke WAIT=exp/Q170_A_DONE bash scripts/q171_targetkv_mc.job.sh >> exp/logs/q171_A.log 2>&1 || { echo "[SMOKE_FAIL]" >> exp/logs/q171_B.log; touch exp/Q171_A_DONE exp/Q171_B_DONE; exit 1; }
GPUS=6,7 JOBS="sand_filling_up_jar:9999 sandcastle_wave_impact:9999" TAG=B WAIT=exp/Q170_B_DONE setsid nohup bash scripts/q171_targetkv_mc.job.sh >> exp/logs/q171_B.log 2>&1 < /dev/null &
GPUS=4,5 JOBS="sand_filling_up_jar:1234 sandcastle_wave_impact:1234" TAG=A bash scripts/q171_targetkv_mc.job.sh >> exp/logs/q171_A.log 2>&1
