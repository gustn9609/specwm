#!/bin/bash
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=specwm_out/latency_unify; mkdir -p $R; CK=$PWD/lingbot-world-v2-14b-causal-fast; A="--action-root $SCRATCH/rt2_action_05 --prompt-file $SCRATCH/p_05.txt --scene 05 --frames 161 --t5-gpu --arms fast_bf16"
python -m specwm.latency_v2 --ckpt $CK $A --areas 399360 --nodes 0,3 --out $R
python -m specwm.latency_v2 --ckpt $CK $A --areas 73728 --out $R
python -m specwm.latency_v2 --ckpt $PWD/lingbot-world-v2-1.3b-causal-fast --cfg i2v-1.3B $A --areas 399360 --out $R/p13
grep -h "^\[fast_bf16" $R/latency_v2.json 2>/dev/null; touch $SCRATCH/exp/Q150_DONE; echo "[Q150_DONE] $(date)"
