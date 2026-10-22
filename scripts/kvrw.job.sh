#!/bin/bash
true
export SPECWM_NO_FLASH=1; export OMP_NUM_THREADS=4; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out
for PV in e3c_repaint_laundered e3a_repaint_fast e3sep_self_mask; do
  echo "##### KV rewrite provider = $PV #####"
  python -m specwm.resync_v3 --ckpt $PWD/lingbot-world-v2-14b-causal-fast --action-root $SCRATCH/rt2_action_05 --prompt-file $SCRATCH/p_05.txt --scene 05 --frames 161 \
    --seeds 1234 --base-video $R/revisit/$PV/video.mp4 --configs W18_sink6 --out $R/kvrw_$PV
done
echo "[KVRW_DONE]"
