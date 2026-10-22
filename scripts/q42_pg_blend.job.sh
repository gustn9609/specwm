#!/bin/bash
# GPU5 (Q41 후) E1 융합=전노드 블렌딩 (테이블 기반, first-write) α 0.2/0.3 x 3시드 + random-address blend 2시드 -> geo
true
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
for i in $(seq 1 240); do [ -f $SCRATCH/exp/Q41_DONE ] && break; sleep 30; done
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-v2-14b-causal-fast; O=$R/posegram/S05_W18
run() { echo "########## PG $* ($(date)) ##########"; python -m specwm.posegram --ckpt $CK --action-root $SCRATCH/traj55_action_05 --prompt-file $SCRATCH/p_05.txt --scene 05 --frames 873 --window 18 --sink 18 --out $O "$@" || echo "!!! FAILED $*"; }
for SD in 1234 4321 9999; do run --mode posegram_blend --alpha 0.3 --write first --seed $SD; run --mode posegram_blend --alpha 0.2 --write first --seed $SD; done
for SD in 1234 4321; do run --mode posegram_rand_blend --alpha 0.3 --write first --seed $SD; done
python $SCRATCH/scripts/geo55.py $R/geo55.json --scan; touch $SCRATCH/exp/Q42_DONE; echo "[Q42_DONE] $(date)"
