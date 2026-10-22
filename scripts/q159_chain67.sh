#!/bin/bash
# GPU 6+7 chain (GPU 2/3 were taken by hl35496): cost sweep -> night A (s1234 both tasks) -> night C (castle s4321/9999)
cd /home/hk28965/specwm; export SCRATCH=/home/hk28965/specwm
CUDA_VISIBLE_DEVICES=6,7 bash scripts/q155_specv0_cost.job.sh >> exp/logs/q155_specv0_cost.log 2>&1
CUDA_VISIBLE_DEVICES=6,7 SEEDS=1234 TAG=A bash scripts/q156_night.job.sh >> exp/logs/q156_night_A.log 2>&1
CUDA_VISIBLE_DEVICES=6,7 SEEDS="4321 9999" TASKS=sandcastle_wave_impact TAG=C bash scripts/q156_night.job.sh >> exp/logs/q156_night_C.log 2>&1
