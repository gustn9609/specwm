#!/bin/bash
# Hand-over at the switch: KV-rewrite (blend6+u2 target where available, else E1' target) but chunks 6.. start from Fast's own chunk re-noised (SDEdit) with decreasing anchoring: ho3 / ho32 / ho321 / ho2 / ho33 / ho333. 9 floor combos.
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V2=$S/lingbot-world-v2; O=$R/e1repro; cd $V2; export PYTHONPATH=$V2
for SC in 05 01 02; do for SD in 1234 4321 9999; do
  T=$O/S$SC/target_s$SD/video.mp4
  echo "########## HANDOVER S$SC s$SD ($(date)) ##########"
  python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast --action-root $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --scene $SC --frames 161 --seeds $SD --base-video $R/floor/base/S${SC}_s$SD/video.mp4 --configs W18_sink6 \
    --arms kvrw_laundered_ho3,kvrw_laundered_ho2,kvrw_laundered_ho32,kvrw_laundered_ho321,kvrw_laundered_ho33,kvrw_laundered_ho333 --kv-providers "laundered=$T" --out $O/S$SC 2>&1 | grep -E "^\[|Error|Traceback"
done; done
touch $S/exp/Q137_DONE; echo "[Q137_DONE] $(date)"
