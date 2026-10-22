#!/bin/bash
# Exp 7 (4090 proxy; no 4090 host reachable from spi0): (1) 360p 2-step, bf16 vs ours(GPTQ int6 + kv4 + tiled VAE) fps/peak on RTX PRO 6000;
# (2) same 'ours' arm under a 24 GB allocator cap (4090 memory emulation) -> fits or OOM; (3) analytic 4090 projection written by the report step.
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=specwm_out/s7_4090proxy; mkdir -p $R; CK=$PWD/lingbot-world-v2-14b-causal-fast
A="--ckpt $CK --action-root $SCRATCH/rt2_action_05 --prompt-file $SCRATCH/p_05.txt --scene 05 --frames 161 --t5-gpu --areas 184320 --nodes 0,3"
echo "########## 4090 PROXY uncapped ($(date)) ##########"; python -m specwm.latency_v2 $A --arms fast_bf16,ours --out $R/uncapped
echo "########## 4090 PROXY 24GB cap, ours ($(date)) ##########"; SPECWM_MEM_CAP_GB=24 python -m specwm.latency_v2 $A --arms ours --out $R/cap24
echo "########## 4090 PROXY 24GB cap, ours, no T5 on GPU ($(date)) ##########"; SPECWM_MEM_CAP_GB=24 python -m specwm.latency_v2 --ckpt $CK --action-root $SCRATCH/rt2_action_05 --prompt-file $SCRATCH/p_05.txt --scene 05 --frames 161 --areas 184320 --nodes 0,3 --arms ours --out $R/cap24_t5cpu
touch $SCRATCH/exp/Q123_DONE; echo "[Q123_DONE] $(date)"
