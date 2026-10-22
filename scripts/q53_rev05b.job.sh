#!/bin/bash
# 재방문 통제 궤적 실험: scene 05, R in 4 8, seeds 1234 4321 — ctrl / posegram_blend α0.3 first / rand_blend
true
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-v2-14b-causal-fast
for RV in 4 8; do for SD in 1234 4321; do for M in "ctrl" "posegram_blend --alpha 0.3 --write first" "posegram_rand_blend --alpha 0.3 --write first"; do
  echo "########## REV S05 r$RV seed $SD mode $M ($(date)) ##########"
  python -m specwm.posegram --ckpt $CK --action-root $SCRATCH/data/rev_action_05_r$RV --prompt-file $SCRATCH/p_05.txt --scene 05 --frames 873 --window 18 --sink 18 \
    --mode $M --seed $SD --out $R/posegram/R05r${RV}_W18 || echo "!!! FAILED r$RV $SD $M"
done; done; done
python $SCRATCH/scripts/geo55.py $R/geo55.json --scan; touch $SCRATCH/exp/q53_rev05b_DONE; echo "[q53_rev05b_DONE] $(date)"
