#!/bin/bash
true
export SPECWM_NO_FLASH=1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD
E=$SCRATCH/epic/G_peel_garlic
python -m specwm.bench_base_official --ckpt $PWD/lingbot-world-base-cam --image $E/image.jpg --action-path $E/action --prompt-file $E/prompt.txt --frames 161 --steps 70 --seed 1234 --out $SCRATCH/lingbot-world-v2/specwm_out/epic/G_peel_garlic_base70
echo "[EPIC_BASE_DONE] G_peel_garlic"
