#!/bin/bash
true
export SPECWM_NO_FLASH=1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD
for W in F_cut_tomato G_peel_garlic; do
  E=$SCRATCH/epic/$W; echo "##### $W #####"
  python -m specwm.bench_base_official --ckpt $PWD/lingbot-world-base-cam --image $E/image.jpg --action-path $E/action --prompt-file $E/prompt.txt --frames 161 --steps 70 --seed 1234 --out $SCRATCH/lingbot-world-v2/specwm_out/epic/${W}_base70
  echo "[EPIC_BASE_DONE] $W"
done
