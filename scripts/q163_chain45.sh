#!/bin/bash
# GPU 4+5: finish hysteresis (castle run3) -> warm-up gating on jar s1234/4321/9999 + castle s1234
cd /home/hk28965/specwm; export SCRATCH=/home/hk28965/specwm
bash scripts/q161_hyst.job.sh >> exp/logs/q161_hyst.log 2>&1
GPUS=4,5 TASKS="sand_filling_up_jar sandcastle_wave_impact" SEEDS="1234" TAG=A bash scripts/q162_warmup.job.sh >> exp/logs/q162_warmup_A.log 2>&1
GPUS=4,5 TASKS="sand_filling_up_jar" SEEDS="4321 9999" TAG=B bash scripts/q162_warmup.job.sh >> exp/logs/q162_warmup_B.log 2>&1
