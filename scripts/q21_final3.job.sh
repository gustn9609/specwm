#!/bin/bash
true
for i in $(seq 1 720); do [ -f $SCRATCH/exp/Q19_DONE ] && [ -f $SCRATCH/exp/Q20_DONE ] && break; sleep 60; done
source $SCRATCH/venv/bin/activate; export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
$SCRATCH/vr-venv/bin/python $SCRATCH/scripts/vr_daemon.py --poll 60 --max-hours 1 --roots floor,handoff_grid,ctx_grid,revisit,repro,demo,handoff
python $SCRATCH/scripts/floor_analysis.py && python $SCRATCH/scripts/demo_gallery.py && python $SCRATCH/scripts/report_all.py
touch $SCRATCH/exp/Q21_DONE; echo "[Q21_DONE] $(date)"
