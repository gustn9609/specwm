#!/bin/bash
# GPU7 (Q37 후): posegram 핵심 arm 시드 9999 (S05) + S01 에서 ctrl/posegram/oracle_kv -> geo scan
true
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-v2-14b-causal-fast
run() { SC=$1; shift; echo "########## PG S$SC $* ($(date)) ##########"; python -m specwm.posegram --ckpt $CK --action-root $SCRATCH/traj55_action_$SC --prompt-file $SCRATCH/p_$SC.txt --scene $SC --frames 873 --window 18 --sink 18 --out $R/posegram/S${SC}_W18 "$@" || echo "!!! FAILED $*"; }
run 05 --mode ctrl --seed 9999
run 05 --mode posegram --write first --theta0 10 --rho0 0.05 --seed 9999
run 05 --mode posegram --write always --theta0 10 --rho0 0.05 --seed 9999
run 05 --mode posegram_rand --write always --theta0 10 --rho0 0.05 --seed 9999
run 05 --mode oracle_kv --seed 9999
for SD in 1234 4321; do run 01 --mode ctrl --seed $SD; run 01 --mode posegram --write first --theta0 10 --rho0 0.05 --seed $SD; run 01 --mode oracle_kv --seed $SD; done
python $SCRATCH/scripts/geo55.py $R/geo55.json --scan
touch $SCRATCH/exp/Q40_DONE; echo "[Q40_DONE] $(date)"
