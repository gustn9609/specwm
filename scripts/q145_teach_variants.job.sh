#!/bin/bash
# E-A variants (360p): mixing alpha, later nodes (low expert), RePaint-style prefix input. COMBOS per shard.
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; O=$SCRATCH/lingbot-world-v2/specwm_out/teach
for CS in $COMBOS; do SC=${CS%_*}; SD=${CS#*_}
  for VAR in "mix05_n0 --nodes 0 --alpha 0.5" "n2 --nodes 2" "n3 --nodes 3" "n23 --nodes 2,3" "ren_n0 --nodes 0 --prefix-mode renoise" "ren_n2 --nodes 2 --prefix-mode renoise" "ren_n23 --nodes 2,3 --prefix-mode renoise"; do set -- $VAR; NAME=$1; shift; D=$O/S$SC/${NAME}_s$SD
    [ -f $D/video.mp4 ] && continue; echo "########## TEACHV $NAME S$SC s$SD ($(date)) ##########"; python -m specwm.teach_node --scene $SC --seed $SD --max-area 184320 --out $D "$@" 2>&1 | grep -E "^\[TEACH|Error|Traceback|SystemExit"
  done
done
touch $SCRATCH/exp/Q145_${SHARD}_DONE; echo "[Q145_DONE shard $SHARD] $(date)"
