#!/bin/bash
# GPU5 (Q48 후): 남은 posegram 영상 VR 채점 (α0.3/0.4 pose 블렌딩 포함) -> 마무리
true
for i in $(seq 1 240); do [ -f $SCRATCH/exp/Q48_DONE ] && break; sleep 30; done
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
$SCRATCH/vr-venv/bin/python $SCRATCH/scripts/vr_daemon.py --poll 30 --max-hours 1.5 --roots posegram
touch $SCRATCH/exp/Q49_DONE; echo "[Q49_DONE] $(date)"
