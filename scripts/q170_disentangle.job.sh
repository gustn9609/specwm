#!/bin/bash
# why did 360p draft + window-3 look better than 480p window-3 (full check)? env: GPUS, JOBS="task:seed ...", WAIT flag
#  (2) check noise level: 480p draft, window-3, full check on the state after 3 of 4 steps (sigma~0.62, like the 2-step 360p draft) -> win3_full_n3
#  (3) number of corrections: 480p window-3, full check at sigma~0.94, threshold = 90th percentile (fewer rejections) -> win3_full_q90
cd /home/hk28965/specwm; export SCRATCH=/home/hk28965/specwm; while [ ! -f $WAIT ]; do sleep 120; done
source venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8; cd lingbot-world-v2; export PYTHONPATH=$PWD
PK="$SCRATCH/venv/bin/python $SCRATCH/scripts/spec_v0_pick.py"; V0=$SCRATCH/lingbot-world-v2/specwm_out/spec_v0
run() { D=$V0/$1/$3_s$2; [ -f $D/video.mp4 ] && return 0; echo "########## SPECV0 $1 $3 s$2 ($(date)) ##########"
  CUDA_VISIBLE_DEVICES=$GPUS python -m specwm.spec_v0 --task $1 --seed $2 --policy $4 "${@:5}" --out $D 2>&1 | grep -E "^\[SPEC|^\[window|Error|Traceback|OutOfMemory|line [0-9]"; }
for J in $JOBS; do T=${J%:*}; SD=${J#*:}
  run $T $SD probe2n3 probe2:3 --chk-list full,g1
  run $T $SD win3_full_n3 window:3:$($PK thr2 $T $SD full 0.5 probe2n3) --chk full --check-at 3 --warmup 2 --steps 16
  run $T $SD win3_full_q90 window:3:$($PK thr2 $T $SD full 0.9) --chk full --warmup 2 --steps 16
done
touch $SCRATCH/exp/Q170_${TAG}_DONE; echo "[Q170_DONE $TAG] $(date)"
