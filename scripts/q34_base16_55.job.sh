#!/bin/bash
true
SC=05 STEPS=16 MA=122880 bash $SCRATCH/scripts/q30_base55.job.sh
source $SCRATCH/venv/bin/activate; python $SCRATCH/scripts/geo55.py $SCRATCH/lingbot-world-v2/specwm_out/geo55.json --scan
touch $SCRATCH/exp/Q34_DONE; echo "[Q34_DONE] $(date)"
