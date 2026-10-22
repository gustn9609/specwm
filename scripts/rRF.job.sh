#!/bin/bash
true
export SPECWM_NO_FLASH=1; export OMP_NUM_THREADS=4; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V2=$S/lingbot-world-v2
CKF=$V2/lingbot-world-v2-14b-causal-fast; PR=$S/p_05.txt; ACT=$S/traj55_action_05; OUT=$R/rollforce; mkdir -p $OUT
# Rolling-Forcing 식 파이프라인이 Fast 의 55초 드리프트를 줄이나. 시드 3개 x {sequential, lag1, lag2}. e2e control 과 같은 W36 sink18.
for SD in 1234 4321 9999; do
  for CFG in "sequential 1 seq" "pipeline 1 lag1" "pipeline 2 lag2"; do
    set -- $CFG; M=$1; LAG=$2; TAG=$3; D=$OUT/${TAG}_s$SD
    [ -f $D/frames_all.npy ] && { echo "##### skip $TAG s$SD #####"; continue; }
    echo "##### $TAG seed $SD #####"
    cd $V2; PYTHONPATH=$V2 python -m specwm.fast_pipeline --ckpt $CKF --action-root $ACT --prompt-file $PR --scene 05 \
      --frames 873 --seed $SD --window 36 --sink 18 --mode $M --lag $LAG --save-every 4 --out $D || echo "!!! $TAG s$SD FAILED"
    sleep 3
  done
done
echo "===== SUMMARY ====="; for SD in 1234 4321 9999; do for T in seq lag1 lag2; do echo "$T s$SD: $(cat $OUT/${T}_s$SD/pipeline.json 2>/dev/null | tr -d '\n ' | cut -c1-160)"; done; done
echo "[RRF_DONE]"
