#!/bin/bash
# PoseGram §7.1: Base 70-step 55초(873프레임) baseline. 사용: SC=05 (기본) / SC=01. GPU 1장 전용, 수 시간.
true
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SC=${SC:-05}; STEPS=${STEPS:-70}; MA=${MA:-184320}; cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD
D=$SCRATCH/lingbot-world-v2/specwm_out/base55/S${SC}_s${STEPS}_a${MA}_s1234
[ -f $D/revisit.json ] || { echo "########## BASE ${STEPS}-step 55s S$SC ($(date)) ##########"
  python -m specwm.bench_base_official --ckpt $PWD/lingbot-world-base-cam --image $SCRATCH/lingbot-world-v2/examples/$SC/image.jpg \
    --action-path $SCRATCH/traj55_action_$SC --prompt-file $SCRATCH/p_$SC.txt --frames 873 --steps $STEPS --max-area $MA --seed 1234 --out $D; }
touch $SCRATCH/exp/Q30_S${SC}_s${STEPS}_a${MA}_DONE; echo "[Q30_DONE S$SC steps$STEPS] $(date)"
