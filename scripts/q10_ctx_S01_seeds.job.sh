#!/bin/bash
# GPU7 (Q7 종료 후): ctx_grid S01 다중 시드 (플로어 부착용)
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD
R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-v2-14b-causal-fast
for i in $(seq 1 300); do [ -f $SCRATCH/exp/Q7_DONE ] && break; sleep 60; done
echo "########## CTX GRID S01 seeds 4321,9999 ($(date)) ##########"
python -m specwm.ctx_grid --ckpt $CK --action-root $SCRATCH/rt2_action_01 --prompt-file $SCRATCH/p_01.txt --scene 01 --frames 161 \
  --base-video $R/floor/base/S01_s1234/video.mp4 --res 1.0,0.75,0.5 --bits 16,8,4 --sigmas 0,0.625,0.833 --seeds 4321,9999 --out $R/ctx_grid/S01
touch $SCRATCH/exp/Q10_DONE; echo "[Q10_DONE] $(date)"
