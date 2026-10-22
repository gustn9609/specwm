#!/bin/bash
# SR-D: 360p 2-step 재생성(동일 시드 → 동일 latent) + latent 1.5x 보간 후 VAE 디코드 (464x848). 85 tasks.
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; cd $S/lingbot-world-v2; export PYTHONPATH=$PWD; CK=$PWD/lingbot-world-v2-14b-causal-fast
echo "########## SR-D latent-up ($(date)) ##########"
python -m specwm.stevo_gen --ckpt $CK --tasks $S/data/stevo/tasks_camera.json --out $R/stevo/lingbot_fast_360_2step_latentup --arm fast_bf16 --max-area 184320 --nodes 0,3 --frames 253 --t5-gpu --latent-up 464,848
touch $S/exp/Q128_DONE; echo "[Q128_DONE] $(date)"
