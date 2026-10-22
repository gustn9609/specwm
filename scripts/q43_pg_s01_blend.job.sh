#!/bin/bash
# GPU7 (Q40 후): S01 에서 oracle 전노드 블렌딩 (일반성) + posegram_blend -> geo
true
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
for i in $(seq 1 240); do [ -f $SCRATCH/exp/Q40_DONE ] && break; sleep 30; done
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-v2-14b-causal-fast; O=$R/posegram/S01_W18
run() { echo "########## PG S01 $* ($(date)) ##########"; python -m specwm.posegram --ckpt $CK --action-root $SCRATCH/traj55_action_01 --prompt-file $SCRATCH/p_01.txt --scene 01 --frames 873 --window 18 --sink 18 --out $O "$@" || echo "!!! FAILED $*"; }
for SD in 1234 4321; do run --mode oracle_blend_all --alpha 0.3 --seed $SD; run --mode posegram_blend --alpha 0.3 --write first --seed $SD; done
run --mode oracle_blend_all --alpha 0.5 --seed 1234; run --mode posegram_rand_blend --alpha 0.3 --write first --seed 1234
python $SCRATCH/scripts/geo55.py $R/geo55.json --scan; touch $SCRATCH/exp/Q43_DONE; echo "[Q43_DONE] $(date)"
