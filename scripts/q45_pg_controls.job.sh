#!/bin/bash
# GPU7 (Q43 후): pose 특이성 분리 대조 — anchor(첫 청크) 블렌딩 / recent(window 밖 최근) 블렌딩, S05 2시드 + S01 1시드; S01 rand blend 시드 4321 -> geo
true
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-v2-14b-causal-fast
run() { SC=$1; shift; echo "########## PG S$SC $* ($(date)) ##########"; python -m specwm.posegram --ckpt $CK --action-root $SCRATCH/traj55_action_$SC --prompt-file $SCRATCH/p_$SC.txt --scene $SC --frames 873 --window 18 --sink 18 --out $R/posegram/S${SC}_W18 "$@" || echo "!!! FAILED $*"; }
for SD in 1234 4321; do run 05 --mode anchor_blend --alpha 0.3 --seed $SD; run 05 --mode recent_blend --alpha 0.3 --seed $SD; done
run 01 --mode posegram_rand_blend --alpha 0.3 --write first --seed 4321; run 01 --mode anchor_blend --alpha 0.3 --seed 1234; run 01 --mode recent_blend --alpha 0.3 --seed 1234
python $SCRATCH/scripts/geo55.py $R/geo55.json --scan; touch $SCRATCH/exp/Q45_DONE; echo "[Q45_DONE] $(date)"
