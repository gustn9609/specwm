#!/bin/bash
# 재방문 통제 궤적 실험: scene 01, R in 0 4, seeds 1234 4321 — ctrl / posegram_blend α0.3 first / rand_blend
true
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-v2-14b-causal-fast
for RV in 0 4; do for SD in 1234 4321; do for M in "ctrl" "posegram_blend --alpha 0.3 --write first" "posegram_rand_blend --alpha 0.3 --write first"; do
  echo "########## REV S01 r$RV seed $SD mode $M ($(date)) ##########"
  python -m specwm.posegram --ckpt $CK --action-root $SCRATCH/data/rev_action_01_r$RV --prompt-file $SCRATCH/p_01.txt --scene 01 --frames 873 --window 18 --sink 18 \
    --mode $M --seed $SD --out $R/posegram/R01r${RV}_W18 || echo "!!! FAILED r$RV $SD $M"
done; done; done
python $SCRATCH/scripts/geo55.py $R/geo55.json --scan; touch $SCRATCH/exp/q54_rev01_DONE; echo "[q54_rev01_DONE] $(date)"
