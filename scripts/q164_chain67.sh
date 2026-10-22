#!/bin/bash
# GPU 6+7: night C (castle s4321/9999: never/probe/spec/every:2/vnode) -> warm-up gating on castle s4321/9999
cd /home/hk28965/specwm; export SCRATCH=/home/hk28965/specwm
CUDA_VISIBLE_DEVICES=6,7 SEEDS="4321 9999" TASKS=sandcastle_wave_impact TAG=C bash scripts/q156_night.job.sh >> exp/logs/q156_night_C.log 2>&1
GPUS=6,7 TASKS="sandcastle_wave_impact" SEEDS="4321 9999" TAG=C bash scripts/q162_warmup.job.sh >> exp/logs/q162_warmup_C.log 2>&1
