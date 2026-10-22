#!/bin/bash
# GPU4 E1: 비파라메트릭 PoseGram 스윕 — write policy x 주소 해상도, random-address, host offload/prefetch 타이밍 -> geo scan
true
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
for i in $(seq 1 120); do [ -f $SCRATCH/exp/PG_SMOKE_OK ] && break; sleep 30; done
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-v2-14b-causal-fast; O=$R/posegram/S05_W18
run() { echo "########## PG $* ($(date)) ##########"; python -m specwm.posegram --ckpt $CK --action-root $SCRATCH/traj55_action_05 --prompt-file $SCRATCH/p_05.txt --scene 05 --frames 873 --window 18 --sink 18 --out $O "$@" || echo "!!! FAILED $*"; }
# (1) write policy (기본 해상도 θ10° ρ0.05), GPU 테이블
for W in always first gated; do run --mode posegram --write $W --theta0 10 --rho0 0.05 --seed 1234; done
# (2) random-address baseline
run --mode posegram_rand --write always --theta0 10 --rho0 0.05 --seed 1234
# (3) 주소 해상도 스윕 (first-write)
run --mode posegram --write first --theta0 5 --rho0 0.02 --seed 1234
run --mode posegram --write first --theta0 20 --rho0 0.1 --seed 1234
run --mode posegram --write first --theta0 10 --rho0 0.02 --seed 1234
run --mode posegram --write first --theta0 5 --rho0 0.05 --seed 1234
# (4) 시스템: 호스트 테이블 + prefetch (E2)
run --mode posegram --write first --theta0 10 --rho0 0.05 --table-device cpu --seed 1234
run --mode posegram --write first --theta0 10 --rho0 0.05 --table-device cpu --prefetch --seed 1234
# (5) 2번째 시드 (핵심 arm)
for W in always first; do run --mode posegram --write $W --theta0 10 --rho0 0.05 --seed 4321; done
run --mode posegram_rand --write always --theta0 10 --rho0 0.05 --seed 4321
python $SCRATCH/scripts/geo55.py $R/geo55.json --scan
touch $SCRATCH/exp/Q39_DONE; echo "[Q39_DONE] $(date)"
