#!/bin/bash
cd /home/hk28965/specwm; while kill -0 537012 2>/dev/null; do sleep 30; done
GPUS=4,5 JOBS="sand_filling_up_jar:1234 sandcastle_wave_impact:1234 sand_filling_up_jar:4321" bash scripts/q169_async.job.sh >> exp/logs/q169_async_45.log 2>&1
touch exp/Q167_DONE; echo "[Q167_DONE] $(date)" >> exp/logs/q169_async_45.log
