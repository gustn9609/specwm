#!/bin/bash
# GPU4 (Q20 후, Q19 VR catch-up 전 빈 시간): Base 플로어 hands seed 31415 (~23분)
true
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD; D=$SCRATCH/lingbot-world-v2/specwm_out/floor/base/Shands_s31415
[ -f $D/revisit.json ] || { echo "########## BASE 70-step Shands seed 31415 ($(date)) ##########"
  python -m specwm.bench_base_official --ckpt $PWD/lingbot-world-base-cam --image $SCRATCH/lingbot-world-v2/examples/hands/image.jpg \
    --action-path $SCRATCH/hands_action --prompt-file $SCRATCH/p_hands1.txt --frames 81 --seed 31415 --out $D; }
touch $SCRATCH/exp/Q24_DONE; echo "[Q24_DONE] $(date)"
