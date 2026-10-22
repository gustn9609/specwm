#!/bin/bash
# whiteboard loop: (1) target KV = block-causal Base with cached clean prefix as the check; (2) stochastic acceptance u vs a=min(1,exp(-beta*dL)).
# env: GPUS, JOBS="task:seed ...", WAIT (flag file), SMOKE=1 -> short clips into _smoke/, TAG
cd /home/hk28965/specwm; export SCRATCH=/home/hk28965/specwm; while [ -n "$WAIT" ] && [ ! -f $WAIT ]; do sleep 120; done
source venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8; cd lingbot-world-v2; export PYTHONPATH=$PWD
PK="$SCRATCH/venv/bin/python $SCRATCH/scripts/spec_v0_pick.py"; V0=$SCRATCH/lingbot-world-v2/specwm_out/spec_v0
run() { D=$V0/$1/$3_s$2; [ -n "$SMOKE" ] && D=$V0/_smoke/$3; [ -f $D/video.mp4 ] && return 0; echo "########## SPECV0 $1 $3 s$2 ($(date)) ##########"
  CUDA_VISIBLE_DEVICES=$GPUS python -m specwm.spec_v0 --task $1 --seed $2 --policy $4 "${@:5}" ${SMOKE:+--frames 61} --out $D 2>&1 | grep -E "^\[SPEC|^\[chunk|Error|Traceback|OutOfMemory|line [0-9]"; }
if [ -n "$SMOKE" ]; then T=sand_filling_up_jar
  run $T 1 kv_probe2 probe2:1 --chk-list full,kv,kvg1
  run $T 1 kv_early early:0.5 --chk kv --warmup 1 --run 1 --steps 4
  run $T 1 mc_probe mc --beta 0 --mc-chk kv --warmup 1 --steps 4
  run $T 1 mc_rej mc --beta 1000 --mc-chk kv --warmup 1 --steps 4
  for n in kv_probe2 kv_early mc_probe mc_rej; do [ -f $V0/_smoke/$n/video.mp4 ] || { echo "[SMOKE_FAIL] $n"; exit 1; }; done; echo "[SMOKE_OK $TAG] $(date)"; exit 0; fi
for J in $JOBS; do T=${J%:*}; SD=${J#*:}
  run $T $SD probe2kv probe2:1 --chk-list full,kv,kvg1
  run $T $SD early_kv early:$($PK thr2 $T $SD kv 0.5 probe2kv) --chk kv --warmup 2 --run 2 --steps 16
  run $T $SD early_kvg1 early:$($PK thr2 $T $SD kvg1 0.5 probe2kv) --chk kvg1 --warmup 2 --run 2 --steps 16
  run $T $SD mcprobe_kv mc --beta 0 --mc-chk kv --warmup 2 --steps 16
  run $T $SD mc_kv mc --beta $($PK beta $T $SD mcprobe_kv) --mc-chk kv --warmup 2 --steps 16
  run $T $SD mcprobe_full mc --beta 0 --mc-chk full --mc-k 1 --warmup 2 --steps 16
  run $T $SD mc_full mc --beta $($PK beta $T $SD mcprobe_full) --mc-chk full --mc-k 1 --warmup 2 --steps 16
done
touch $SCRATCH/exp/Q171_${TAG}_DONE; echo "[Q171_DONE $TAG] $(date)"
