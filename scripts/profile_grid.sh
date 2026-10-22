#!/bin/bash
# one-step Base check cost grid (action item 6): expert x CFG x window x resolution, batch 1, one GPU ($1)
cd /home/hk28965/specwm/lingbot-world-v2; export SPECWM_NO_FLASH=1 PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=$1
for EX in low high; do for CFG in 0 1; do for AREA in 399360 184320; do for P in 3 9 18; do
  timeout 900 ../venv/bin/python -m specwm.onestep_profile --expert $EX --cfg $CFG --area $AREA --prefix $P --batches 1 --tag grid 2>&1 | grep -E "profile\]" | cut -c1-150
done; done; done; done
echo "[profile_grid] done $(date)"
