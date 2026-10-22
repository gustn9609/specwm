#!/bin/bash
# async early-send check + window verification. env: GPUS, JOBS="task:seed ...", SMOKE=1 (short clips into _smoke/)
cd /home/hk28965/specwm; export SCRATCH=/home/hk28965/specwm; source venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8; cd lingbot-world-v2; export PYTHONPATH=$PWD
PK="$SCRATCH/venv/bin/python $SCRATCH/scripts/spec_v0_pick.py"; V0=$SCRATCH/lingbot-world-v2/specwm_out/spec_v0
run() { # run <task> <seed> <name> <policy> <args...>
  D=$V0/$1/$3_s$2; [ -n "$SMOKE" ] && D=$V0/_smoke/$3; [ -f $D/video.mp4 ] && return 0
  echo "########## SPECV0 $1 $3 s$2 ($(date)) ##########"
  CUDA_VISIBLE_DEVICES=$GPUS python -m specwm.spec_v0 --task $1 --seed $2 --policy $4 "${@:5}" ${SMOKE:+--frames 61} --out $D 2>&1 | grep -E "^\[SPEC|^\[chunk|^\[window|Error|Traceback|OutOfMemory|line [0-9]"; }
if [ -n "$SMOKE" ]; then T=sand_filling_up_jar
  run $T 1 early early:0.40 --chk a240g1c3 --warmup 2 --run 2 --steps 4
  run $T 1 win2 window:2:0.40 --chk a240g1c3 --warmup 1 --steps 4
  run $T 1 win2_rej window:2:0.0 --chk a240g1c3 --warmup 1 --steps 4
  run $T 1 early360 early:0.40 --chk g1 --warmup 2 --run 2 --steps 4 --max-area 184320 --nodes 0,3 --corr-area 399360
  for n in early win2 win2_rej early360; do [ -f $V0/_smoke/$n/video.mp4 ] || { echo "[SMOKE_FAIL] $n"; exit 1; }; done; echo "[SMOKE_OK] $(date)"; exit 0; fi
for J in $JOBS; do TASK=${J%:*}; SD=${J#*:}
  for P in a240g1c3; do run $TASK $SD early_$P early:$($PK thr2 $TASK $SD $P 0.5) --chk $P --warmup 2 --run 2 --steps 16; done
  TH=$($PK thr2 $TASK $SD a240g1c3 0.5)
  run $TASK $SD win3_a240g1c3 window:3:$TH --chk a240g1c3 --warmup 2 --steps 16
  run $TASK $SD win3_full window:3:$($PK thr2 $TASK $SD full 0.5) --chk full --warmup 2 --steps 16
  # edge-like draft: Fast 360p 2-step, Base check at 360p, Base correction at 480p
  D=$V0/$TASK/probe2_360_s$SD; [ -f $D/video.mp4 ] || { echo "########## SPECV0 $TASK probe2_360 s$SD ($(date)) ##########"; CUDA_VISIBLE_DEVICES=$GPUS python -m specwm.spec_v0 --task $TASK --seed $SD --policy probe2:1 --chk-list full,g1,g1c9,a240g1c3 --max-area 184320 --nodes 0,3 --out $D 2>&1 | grep -E "^\[SPEC|Error|Traceback|OutOfMemory|line [0-9]"; }
  run $TASK $SD d360_never never --max-area 184320 --nodes 0,3
  run $TASK $SD d360_early_g1 early:$($PK thr2 $TASK $SD g1 0.5 probe2_360) --chk g1 --warmup 2 --run 2 --steps 16 --max-area 184320 --nodes 0,3 --corr-area 399360
  run $TASK $SD d360_win3_g1 window:3:$($PK thr2 $TASK $SD g1 0.5 probe2_360) --chk g1 --warmup 2 --steps 16 --max-area 184320 --nodes 0,3 --corr-area 399360
done
