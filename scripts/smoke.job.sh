#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world
# wait (max 60min) for weights to finish downloading
for i in $(seq 1 60); do
  [ -f "$SCRATCH/lingbot-world/lingbot-world-base-cam/Wan2.1_VAE.pth" ] && break; sleep 60; done
python $SCRATCH/specwm-lingbot/slurm/preflight.py ./lingbot-world-base-cam || exit 1
python -m specwm.l1_stage_a --ckpt ./lingbot-world-base-cam   --image examples/00/image.jpg --action examples/00   --nodes 6 --seeds 1 --frames 41 --quant int8 --pass 1 || exit 1
echo "SMOKE_OK" >> $SCRATCH/specwm_status.txt
