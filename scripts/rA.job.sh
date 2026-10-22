#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1; export OMP_NUM_THREADS=4; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V2=$S/lingbot-world-v2
CKF=$V2/lingbot-world-v2-14b-causal-fast; PR=$S/p_05.txt; ACT=$S/rt2_action_05; OUT=$R/ideasAB; mkdir -p $OUT
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
for M in sequential pipeline; do
  if [ ! -f $OUT/A_$M/frames_all.npy ]; then
    echo "##### A $M #####"
    cd $V2; PYTHONPATH=$V2 python -m specwm.fast_pipeline --ckpt $CKF --action-root $ACT --prompt-file $PR --scene 05 \
      --frames 161 --seed 1234 --window 18 --sink 6 --mode $M $([ $M = pipeline ] && echo --time-batch) --out $OUT/A_$M || echo "!!! A $M FAILED"
    sleep 5
  fi
done
echo "===== SUMMARY ====="; for d in A_sequential A_pipeline; do echo "$d: $(cat $OUT/$d/pipeline.json 2>/dev/null | tr -d '\n ' | cut -c1-240)"; done
echo "[RA_DONE]"
