#!/bin/bash
# GPU7 PoseGram 체인: (1) Fast 55s window sweep W18/36/72 sink18 (baseline 재현) (2) E0 oracle probe (W18 sink18, 2시드) (3) geo probe 전체
true
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-v2-14b-causal-fast
for W in 18 36 72; do D=$R/fast55/S05_W${W}_sink18_s1234
  [ -f $D/frames_all.npy ] && { echo "[skip] $D"; continue; }
  echo "########## FAST55 W$W sink18 ($(date)) ##########"
  python -m specwm.e2e_stage --ckpt $CK --action-root $SCRATCH/traj55_action_05 --prompt-file $SCRATCH/p_05.txt --scene 05 --frames 873 --seed 1234 \
    --window $W --sink 18 --state $D --gen-until 73 --anchor fixed --decode || echo "!!! W$W FAILED"
done
echo "########## ORACLE PROBE S05 ($(date)) ##########"
python -m specwm.oracle_probe --ckpt $CK --action-root $SCRATCH/traj55_action_05 --prompt-file $SCRATCH/p_05.txt --scene 05 --frames 873 --window 18 --sink 18 \
  --seeds 1234,4321 --alphas 0.1,0.3,0.5 --rand-alpha 0.3 --out $R/oracle/S05_W18
echo "########## GEO55 ($(date)) ##########"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python $SCRATCH/scripts/geo55.py $R/geo55.json --scan
touch $SCRATCH/exp/Q32_DONE; echo "[Q32_DONE] $(date)"
