#!/bin/bash
# GPU4 (Q39 후): posegram_blend α0.1 2시드, 고해상도 주소(θ5 ρ0.02) blend, write=always blend, prefetch+cpu blend(시스템) -> geo
true
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
for i in $(seq 1 240); do [ -f $SCRATCH/exp/Q39_DONE ] && break; sleep 30; done
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-v2-14b-causal-fast; O=$R/posegram/S05_W18
run() { echo "########## PG $* ($(date)) ##########"; python -m specwm.posegram --ckpt $CK --action-root $SCRATCH/traj55_action_05 --prompt-file $SCRATCH/p_05.txt --scene 05 --frames 873 --window 18 --sink 18 --out $O "$@" || echo "!!! FAILED $*"; }
for SD in 1234 4321; do run --mode posegram_blend --alpha 0.1 --write first --seed $SD; done
run --mode posegram_blend --alpha 0.3 --write first --theta0 5 --rho0 0.02 --seed 1234
run --mode posegram_blend --alpha 0.3 --write always --seed 1234
run --mode posegram_blend --alpha 0.3 --write gated --seed 1234
run --mode posegram_blend --alpha 0.3 --write first --table-device cpu --prefetch --seed 1234
python $SCRATCH/scripts/geo55.py $R/geo55.json --scan; touch $SCRATCH/exp/Q44_DONE; echo "[Q44_DONE] $(date)"
