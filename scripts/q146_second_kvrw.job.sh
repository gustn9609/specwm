#!/bin/bash
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V2=$S/lingbot-world-v2; SC=05; O=$R/spec_dist/S$SC; cd $V2; export PYTHONPATH=$V2
for SD in $SEEDS; do TD=$O/target2_s$SD; [ -f $TD/video.mp4 ] || continue; [ -f $O/W18_sink6/kvrw_second_k10_s$SD/video.mp4 ] && continue
  python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast --action-root $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --scene $SC --frames 161 --seeds $SD --base-video $R/floor/base/S${SC}_s$SD/video.mp4 --configs W18_sink6 --arms kvrw_second_k10 --kv-providers "second=$TD/video.mp4" --out $O 2>&1 | grep -E "^\[W18|Error|Traceback"
done; touch $S/exp/Q146_${SHARD}_DONE; echo "[Q146_DONE $SHARD] $(date)"
