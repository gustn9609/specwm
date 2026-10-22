#!/bin/bash
# Exp 7b: pre-quantized-deployment emulation — quantize on the big GPU, then cap the allocator at 24 GB (T5 on CPU / on GPU) for the 360p 2-step 'ours' arm.
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=specwm_out/s7_4090proxy; CK=$PWD/lingbot-world-v2-14b-causal-fast
A="--ckpt $CK --action-root $SCRATCH/rt2_action_05 --prompt-file $SCRATCH/p_05.txt --scene 05 --frames 161 --areas 184320 --nodes 0,3 --arms ours"
echo "########## cap24 after quant, T5 CPU ($(date)) ##########"; SPECWM_MEM_CAP_AFTER_QUANT_GB=24 python -m specwm.latency_v2 $A --out $R/cap24aq_t5cpu
echo "########## cap24 after quant, T5 GPU ($(date)) ##########"; SPECWM_MEM_CAP_AFTER_QUANT_GB=24 python -m specwm.latency_v2 $A --t5-gpu --out $R/cap24aq_t5gpu
echo "########## cap20 after quant, T5 CPU ($(date)) ##########"; SPECWM_MEM_CAP_AFTER_QUANT_GB=20 python -m specwm.latency_v2 $A --out $R/cap20aq_t5cpu
touch $SCRATCH/exp/Q123B_DONE; echo "[Q123B_DONE] $(date)"
