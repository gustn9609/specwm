#!/bin/bash
# GPU7 (Q16 후): ctx_grid S00 다중 시드
true
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
for i in $(seq 1 480); do [ -f $SCRATCH/exp/Q16_DONE ] && break; sleep 60; done
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-v2-14b-causal-fast
echo "########## CTX GRID S00 seeds 4321,9999 ($(date)) ##########"
python -m specwm.ctx_grid --ckpt $CK --action-root $SCRATCH/rt2_action_00 --prompt-file $SCRATCH/p_00.txt --scene 00 --frames 161 \
  --base-video $R/floor/base/S00_s1234/video.mp4 --res 1.0,0.75,0.5 --bits 16,8,4 --sigmas 0,0.625,0.833 --seeds 4321,9999 --out $R/ctx_grid/S00
touch $SCRATCH/exp/Q22_DONE; echo "[Q22_DONE] $(date)"
