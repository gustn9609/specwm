#!/bin/bash
# GPU7: Guandao 질문 — self-forcing(Fast causal) 1프레임 vs 3프레임 청크 디코딩 지연 (480x832, T5 GPU, 161f)
true
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-v2-14b-causal-fast
for CS in 1 3; do echo "########## LATENCY chunk_size=$CS ($(date)) ##########"
  python -m specwm.latency_v2 --ckpt $CK --action-root $SCRATCH/rt2_action_05 --prompt-file $SCRATCH/p_05.txt --scene 05 --frames 161 \
    --arms fast_bf16 --areas 399360,184320 --t5-gpu --chunk-size $CS --steady-from 6 --k 6 --out $R/latency_chunksize; done
touch $SCRATCH/exp/Q50_DONE; echo "[Q50_DONE] $(date)"
