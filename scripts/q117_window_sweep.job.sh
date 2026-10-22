#!/bin/bash
# Exp 2 (recovery side): KV-rewrite window k = 1/2/3 chunks (0.75/1.5/2.25 s) vs k=6 (4.5 s, E1'), ideal (resync_k) 도 같이. e1repro 480p targets 재사용, 3 scenes x 3 seeds (Base 영상은 seed 일치).
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V2=$S/lingbot-world-v2; O=$R/e1repro; cd $V2; export PYTHONPATH=$V2
for SC in 05 01 02; do for SD in 1234 4321 9999; do
  echo "########## WINDOW SWEEP S$SC s$SD ($(date)) ##########"
  python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast --action-root $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --scene $SC --frames 161 --seeds $SD --base-video $R/floor/base/S${SC}_s$SD/video.mp4 --configs W18_sink6 \
    --arms resync_k1,resync_k2,resync_k3,kvrw_laundered_k1,kvrw_laundered_k2,kvrw_laundered_k3 --kv-providers "laundered=$O/S$SC/target_s$SD/video.mp4" --out $O/S$SC
done; done
touch $S/exp/Q117_DONE; echo "[Q117_DONE] $(date)"
