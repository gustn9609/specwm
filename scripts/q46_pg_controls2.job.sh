#!/bin/bash
# GPU6 (Q36 후): pose 특이성 대조 3번째 시드 (S05 9999: rand/anchor/recent 블렌딩) + S01 posegram_blend 9999/ctrl 9999 -> geo
true
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
for i in $(seq 1 240); do [ -f $SCRATCH/exp/Q36_DONE ] && break; sleep 30; done
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-v2-14b-causal-fast
run() { SC=$1; shift; echo "########## PG S$SC $* ($(date)) ##########"; python -m specwm.posegram --ckpt $CK --action-root $SCRATCH/traj55_action_$SC --prompt-file $SCRATCH/p_$SC.txt --scene $SC --frames 873 --window 18 --sink 18 --out $R/posegram/S${SC}_W18 "$@" || echo "!!! FAILED $*"; }
run 05 --mode posegram_rand_blend --alpha 0.3 --write first --seed 9999; run 05 --mode anchor_blend --alpha 0.3 --seed 9999; run 05 --mode recent_blend --alpha 0.3 --seed 9999
run 01 --mode ctrl --seed 9999; run 01 --mode posegram_blend --alpha 0.3 --write first --seed 9999; run 01 --mode posegram_rand_blend --alpha 0.3 --write first --seed 9999
python $SCRATCH/scripts/geo55.py $R/geo55.json --scan; touch $SCRATCH/exp/Q46_DONE; echo "[Q46_DONE] $(date)"
