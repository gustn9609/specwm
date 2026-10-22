#!/bin/bash
# S1: Fast + WorldCache(공식 판정, Fast 이식) fps / peak memory. 480p & 360p x {4-step, 2-step} x {WC off, k3 thr0.06, k3 thr0.10}
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out/s1_fastwc; mkdir -p $R; CK=$PWD/lingbot-world-v2-14b-causal-fast
run() { echo "########## S1 $* ($(date)) ##########"; python -m specwm.latency_v2 --ckpt $CK --action-root $SCRATCH/rt2_action_05 --prompt-file $SCRATCH/p_05.txt --scene 05 --frames 161 --arms fast_bf16 --t5-gpu --out $R "$@"; }
run --areas 399360,184320
run --areas 399360,184320 --fast-wc 3 --wc-thresh 0.06
run --areas 399360,184320 --fast-wc 3 --wc-thresh 0.10
run --areas 184320 --nodes 0,3
run --areas 184320 --nodes 0,3 --fast-wc 3 --wc-thresh 0.06
run --areas 184320 --nodes 0,3 --fast-wc 3 --wc-thresh 0.10
touch $SCRATCH/exp/Q110_DONE; echo "[Q110_DONE] $(date)"
