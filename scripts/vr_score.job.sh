#!/bin/bash

set -x
true
source $SCRATCH/vr-venv/bin/activate
export HF_HOME=$SCRATCH/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd $SCRATCH/VisionReward
nvidia-smi
python3 $SCRATCH/vr_score.py $SCRATCH/vr_entries.json
