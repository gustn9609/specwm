#!/bin/bash
# GPU6: Base70 55s S05 시드 4321, 9999 (@256x464) -> 55s 기하 지표의 Base 시드 플로어; 이후 S01 4321
true
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out
for SD in 4321 9999; do D=$R/base55/S05_s70_a122880_s$SD; [ -f $D/revisit.json ] && continue
  echo "########## BASE70 55s S05 seed $SD ($(date)) ##########"
  python -m specwm.bench_base_official --ckpt $PWD/lingbot-world-base-cam --image $SCRATCH/lingbot-world-v2/examples/05/image.jpg \
    --action-path $SCRATCH/traj55_action_05 --prompt-file $SCRATCH/p_05.txt --frames 873 --steps 70 --max-area 122880 --seed $SD --out $D; done
D=$R/base55/S01_s70_a122880_s4321; [ -f $D/revisit.json ] || { echo "########## BASE70 55s S01 seed 4321 ($(date)) ##########"
  python -m specwm.bench_base_official --ckpt $PWD/lingbot-world-base-cam --image $SCRATCH/lingbot-world-v2/examples/01/image.jpg \
    --action-path $SCRATCH/traj55_action_01 --prompt-file $SCRATCH/p_01.txt --frames 873 --steps 70 --max-area 122880 --seed 4321 --out $D; }
python $SCRATCH/scripts/geo55.py $R/geo55.json --scan
touch $SCRATCH/exp/Q36_DONE; echo "[Q36_DONE] $(date)"
