#!/bin/bash
true
export SPECWM_NO_FLASH=1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD
R=$SCRATCH/lingbot-world-v2/specwm_out
for SD in 1234 9999; do
  echo "##### Base16 20 s, seed $SD #####"
  python -m specwm.bench_base_official --ckpt $PWD/lingbot-world-base-cam --image $SCRATCH/lingbot-world-v2/examples/05/image.jpg --action-path $SCRATCH/traj20_action_05 \
    --prompt-file $SCRATCH/p_05.txt --frames 321 --steps 16 --seed $SD --out $R/revisit/base20_S05_s16_seed$SD
done
echo "[JB_DONE]"
