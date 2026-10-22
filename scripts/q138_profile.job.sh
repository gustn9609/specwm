#!/bin/bash
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; CK=$PWD/lingbot-world-v2-14b-causal-fast; O=specwm_out/profile_fast; mkdir -p $O
for cfg in "399360 0,1,2,3 480p_4step" "184320 0,1,2,3 360p_4step" "184320 0,3 360p_2step" "399360 0,3 480p_2step"; do set -- $cfg
  echo "########## PROFILE $3 ($(date)) ##########"; python -m specwm.profile_fast --ckpt $CK --action-root $SCRATCH/rt2_action_05 --prompt-file $SCRATCH/p_05.txt --scene 05 --max-area $1 --nodes $2 --out $O/$3.json 2>&1 | grep -E "^\[|Error|Traceback"; done
touch $SCRATCH/exp/Q138_DONE; echo "[Q138_DONE] $(date)"
