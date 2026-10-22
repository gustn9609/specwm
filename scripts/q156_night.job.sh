#!/bin/bash
# night orchestrator (GPU pair in CUDA_VISIBLE_DEVICES): verifier probe -> speculative per-token acceptance -> verify_node gating with the cheapest validated correction.
# SEEDS env; runs both tasks. Waits for WAIT flag file first.
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; V=$SCRATCH/lingbot-world-v2/specwm_out/spec_v0
while [ -n "$WAIT" ] && [ ! -f $WAIT ]; do sleep 120; done
run() { # run <task> <name> <policy> <extra args>
  D=$V/$1/$2_s$SD; [ -f $D/video.mp4 ] && return; echo "########## SPECV0 $1 $2 s$SD ($(date)) ##########"
  python -m specwm.spec_v0 --task $1 --seed $SD --policy $3 ${@:4} --out $D 2>&1 | grep -E "^\[SPEC|^\[chunk|Error|Traceback|OutOfMemory"; }
for SD in $SEEDS; do for TASK in ${TASKS:-sand_filling_up_jar sandcastle_wave_impact}; do
  run $TASK never never
  run $TASK probe probe --check-nodes 0,1,2,3
  run $TASK spec_n12 spec --check-nodes 1,2
  run $TASK spec_n123_tau0.5 spec --check-nodes 1,2,3 --tau 0.5
  if [ -f $SCRATCH/exp/Q155_DONE ] && [ ! -f $SCRATCH/exp/Q155_EVAL ]; then $SCRATCH/venv/bin/python $SCRATCH/scripts/spec_v0_eval.py > $SCRATCH/logs/spec_v0_eval.out 2>&1 && touch $SCRATCH/exp/Q155_EVAL; fi
  [ -f $SCRATCH/exp/Q155_DONE ] && CORR=$($SCRATCH/venv/bin/python $SCRATCH/scripts/spec_v0_pick.py corr 2>/dev/null) || CORR=""
  run $TASK every_2 every:2
  for Q in 0.5 0.75; do THR=$($SCRATCH/venv/bin/python $SCRATCH/scripts/spec_v0_pick.py thr $TASK $SD 1 $Q); run $TASK vnode1_q${Q#0.} verify_node:1:$THR $CORR; done
done; done
touch $SCRATCH/exp/Q156_${TAG}_DONE; echo "[Q156_DONE $TAG] $(date)"
