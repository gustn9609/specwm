#!/bin/bash
# wait for both always_s1234 videos, stop the q152 seed jobs (4-7), smoke-test the new policies on 4+5 (short clip), then start night B/C.
cd /home/hk28965/specwm; export SCRATCH=/home/hk28965/specwm; V=lingbot-world-v2/specwm_out/spec_v0
while [ ! -f $V/sand_filling_up_jar/always_s1234/video.mp4 ] || [ ! -f $V/sandcastle_wave_impact/always_s1234/video.mp4 ]; do sleep 120; done
for p in 841346 841347; do kill $p 2>/dev/null; done; sleep 2
for p in $(ps -eo pid,args | grep "specwm.spec_v0 --task" | grep -v grep | grep -E "seed (4321|9999)" | awk '{print $1}'); do kill $p; done; sleep 10
source venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8; cd lingbot-world-v2; export PYTHONPATH=$PWD
for spec in "probe|probe --check-nodes 0,1,2,3" "spec_n12|spec --check-nodes 1,2" "spec_n123|spec --check-nodes 1,2,3 --tau 0.5" "vnode|verify_node:2:0.3 --steps 4 --ctx 9"; do N=${spec%%|*}; A=${spec#*|}
  echo "########## SMOKE $N ($(date)) ##########"; CUDA_VISIBLE_DEVICES=4,5 python -m specwm.spec_v0 --task sand_filling_up_jar --seed 1 --frames 37 --policy ${A%% *} ${A#* } --out $SCRATCH/$V/_smoke/$N 2>&1 | grep -E "^\[SPEC|^\[chunk|Error|Traceback|OutOfMemory|line "; done
cd $SCRATCH; touch exp/Q152_KILLED; echo "[SMOKE_DONE] $(date)"
CUDA_VISIBLE_DEVICES=4,5 SEEDS="4321 9999" TASKS=sand_filling_up_jar TAG=B nohup bash scripts/q156_night.job.sh > exp/logs/q156_night_B.log 2>&1 &
