#!/bin/bash
cd /home/hk28965/specwm; while kill -0 537216 2>/dev/null; do sleep 30; done
while ! grep -q "SMOKE_OK\|SMOKE_FAIL" exp/logs/q169_async_45.log 2>/dev/null; do sleep 60; done
grep -q SMOKE_OK exp/logs/q169_async_45.log && GPUS=6,7 JOBS="sand_filling_up_jar:9999 sandcastle_wave_impact:4321 sandcastle_wave_impact:9999" bash scripts/q169_async.job.sh >> exp/logs/q169_async_67.log 2>&1
source venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8; cd lingbot-world-v2; export PYTHONPATH=$PWD; V=/home/hk28965/specwm/lingbot-world-v2/specwm_out/spec_v0/sandcastle_wave_impact
for P in never every:2; do D=$V/${P/:/_}_s9999; [ -f $D/video.mp4 ] || CUDA_VISIBLE_DEVICES=6,7 python -m specwm.spec_v0 --task sandcastle_wave_impact --seed 9999 --policy $P --out $D 2>&1 | grep -E "^\[SPEC|Error|Traceback|OutOfMemory" >> /home/hk28965/specwm/exp/logs/q169_async_67.log; done
cd /home/hk28965/specwm; touch exp/Q168_DONE; echo "[Q168_DONE] $(date)" >> exp/logs/q169_async_67.log
