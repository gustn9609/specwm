#!/bin/bash
# P1: anchor restart on 85 StEvo tasks (shard via SHARD=i/n)
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD
python $SCRATCH/scripts/c2_anchor_stevo.py --shard $SHARD 2>&1 | grep -E "^\[|Error|Traceback|line "
touch $SCRATCH/exp/Q133_${SHARD%/*}_DONE; echo "[Q133_DONE shard $SHARD] $(date)"
