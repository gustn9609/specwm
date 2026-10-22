#!/bin/bash
# E-A pilot (360p: Fast + one Base expert fit in 96 GB): teacher-corrected coarse node, 9 floor combos x variants: n0 (node 0 every chunk), n01 (nodes 0+1), n0_e2 (node 0 every 2nd chunk), self (control)
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; O=$SCRATCH/lingbot-world-v2/specwm_out/teach
for CS in "05 1234" "05 4321" "05 9999" "01 1234" "01 4321" "01 9999" "02 1234" "02 4321" "02 9999"; do set -- $CS; SC=$1; SD=$2
  for VAR in "self --self-teach --nodes 0" "n0 --nodes 0" "n01 --nodes 0,1" "n0_e2 --nodes 0 --every 2"; do set -- $VAR; NAME=$1; shift; D=$O/S$SC/${NAME}_s$SD
    [ -f $D/video.mp4 ] && continue; echo "########## TEACH $NAME S$SC s$SD ($(date)) ##########"; python -m specwm.teach_node --scene $SC --seed $SD --max-area 184320 --out $D "$@" 2>&1 | grep -E "^\[|Error|Traceback|SystemExit"
  done
done
touch $SCRATCH/exp/Q144_DONE; echo "[Q144_DONE] $(date)"
