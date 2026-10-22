#!/bin/bash
true
export SPECWM_NO_FLASH=1; export OMP_NUM_THREADS=4; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V2=$S/lingbot-world-v2; CKF=$V2/lingbot-world-v2-14b-causal-fast; PR=$S/p_05.txt; ACT=$S/traj55_action_05; OUT=$R/winsweep; mkdir -p $OUT
for SD in 1234 4321 9999; do for CFG in "54 18" "54 36"; do set -- $CFG; W=$1; SK=$2; D=$OUT/W${W}_sink${SK}_s$SD
  [ -f $D/frames_all.npy ] && { echo "##### skip W$W sink$SK s$SD #####"; continue; }; echo "##### W$W sink$SK seed $SD #####"; mkdir -p $D
  cd $V2; PYTHONPATH=$V2 python -m specwm.e2e_stage --ckpt $CKF --action-root $ACT --prompt-file $PR --scene 05 --frames 873 --seed $SD --window $W --sink $SK --state $D --gen-until 73 --anchor fixed --decode || echo "!!! W$W sink$SK s$SD FAILED"
done; done
echo "===== SUMMARY ====="; for d in $OUT/*/; do echo "$(basename $d): $([ -f $d/frames_all.npy ] && echo OK || echo MISSING)"; done; echo "[RWIN_DONE]"
