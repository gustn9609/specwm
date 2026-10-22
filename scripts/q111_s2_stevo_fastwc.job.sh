#!/bin/bash
# S2: StEvo 85 camera tasks, Fast 360p + 2-step(nodes 0,3) + WorldCache k3 thr0.06 (2-step 에서는 drift ~1.17 >> thr 이라 skip 0 = 360p+2-step 과 동일; 이후 tuned-thr 변형은 별도 arm) → judge
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-v2-14b-causal-fast
echo "########## S2 STEVO fast 360p 2-step WC ($(date)) ##########"
python -m specwm.stevo_gen --ckpt $CK --tasks $SCRATCH/data/stevo/tasks_camera.json --out $R/stevo/lingbot_fast_360_2step --arm fast_bf16 --max-area 184320 --nodes 0,3 --wc 3 --wc-thresh 0.06 --frames 253 --t5-gpu
touch $SCRATCH/exp/Q111_DONE; echo "[Q111_DONE] $(date)"
