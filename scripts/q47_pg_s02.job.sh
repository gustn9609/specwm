#!/bin/bash
# GPU4: 3번째 씬 S02 (urban) — ctrl / posegram_blend / rand_blend x 2시드 -> 일반성
true
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-v2-14b-causal-fast
run() { echo "########## PG S02 $* ($(date)) ##########"; python -m specwm.posegram --ckpt $CK --action-root $SCRATCH/traj55_action_02 --prompt-file $SCRATCH/p_02.txt --scene 02 --frames 873 --window 18 --sink 18 --out $R/posegram/S02_W18 "$@" || echo "!!! FAILED $*"; }
for SD in 1234 4321; do run --mode ctrl --seed $SD; run --mode posegram_blend --alpha 0.3 --write first --seed $SD; run --mode posegram_rand_blend --alpha 0.3 --write first --seed $SD; done
python $SCRATCH/scripts/geo55.py $R/geo55.json --scan; touch $SCRATCH/exp/Q47_DONE; echo "[Q47_DONE] $(date)"
