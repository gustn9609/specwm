#!/bin/bash
# queue worker: pops bundles from exp/queue/bundles.txt (flock) and runs them on GPUS. WAITPID: start after this pid exits. FREEWAIT=1: start only once both GPUs are free.
cd /home/hk28965/specwm; source venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
V0=/home/hk28965/specwm/lingbot-world-v2/specwm_out/spec_v0; Q=${Q:-exp/queue/bundles.txt}
while [ -n "$WAITPID" ] && kill -0 $WAITPID 2>/dev/null; do sleep 30; done
free() { for gid in ${GPUS//,/ }; do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gid); [ $m -lt 2000 ] || return 1; done; return 0; }
run() { D=$V0/$1/$3_s$2; [ -f $D/video.mp4 ] && return 0; ps -eo args | grep -F -- "--out $D" | grep -qv grep && { echo "[skip $3 s$2: already running]"; return 0; }; echo "########## [$GPUS] SPECV0 $1 $3 s$2 ($(date)) ##########"
  (cd lingbot-world-v2; PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=$GPUS python -m specwm.spec_v0 --task $1 --seed $2 --policy $4 "${@:5}" --out $D 2>&1 | grep -E "^\[SPEC|^\[chunk0|^\[window|Error|Traceback|OutOfMemory|line [0-9]"); [ -f $D/video.mp4 ]; }
source scripts/q173_bundles.sh
while true; do
  if [ -n "$FREEWAIT" ]; then until free; do sleep 120; done; fi
  B=$(flock $Q.lock -c "head -1 $Q; sed -i 1d $Q"); [ -z "$B" ] && break
  echo "=== [$GPUS] bundle $B ($(date)) ==="; bundle $B; echo "=== [$GPUS] bundle $B done ($(date)) ==="
done
echo "[WORKER_DONE $GPUS] $(date)"
