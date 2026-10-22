#!/bin/bash
# GPU7: Fast55 baseline 보강 - W36 시드 4321/9999, S01 W18 3시드 (Base55 S01 과 비교), geo scan
true
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-v2-14b-causal-fast
run() { # SC W SD
  D=$R/fast55/S$1_W$2_sink18_s$3; [ -f $D/frames_all.npy ] && return
  echo "########## FAST55 S$1 W$2 seed $3 ($(date)) ##########"
  python -m specwm.e2e_stage --ckpt $CK --action-root $SCRATCH/traj55_action_$1 --prompt-file $SCRATCH/p_$1.txt --scene $1 --frames 873 --seed $3 \
    --window $2 --sink 18 --state $D --gen-until 73 --anchor fixed --decode || echo "!!! S$1 W$2 s$3 FAILED"; }
run 05 36 4321; run 05 36 9999; run 01 18 1234; run 01 18 4321; run 01 18 9999
python $SCRATCH/scripts/geo55.py $R/geo55.json --scan
touch $SCRATCH/exp/Q37_DONE; echo "[Q37_DONE] $(date)"
