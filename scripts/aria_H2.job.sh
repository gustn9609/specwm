#!/bin/bash
true
export SPECWM_NO_FLASH=1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD
for W in D_wash_tomato B2_pour_liquid_pan; do for SD in 1234 42; do E=$SCRATCH/aria/win_$W; echo "##### $W 81f seed $SD #####"
  python -m specwm.bench_base_official --ckpt $PWD/lingbot-world-base-cam --image $E/image.jpg --action-path $E/action --prompt-file $E/prompt.txt --frames 81 --steps 70 --seed $SD --out $SCRATCH/lingbot-world-v2/specwm_out/aria/${W}_base70_81f_s$SD
  echo "[ARIA_BASE_DONE] $W s$SD"
done; done
