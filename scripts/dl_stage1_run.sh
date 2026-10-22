#!/bin/bash
S=$SCRATCH
true
source $S/specwm-venv/bin/activate
export HF_HUB_DISABLE_XET=1
cd $S && python dl_stage1.py > $S/dl_stage1.log 2>&1
