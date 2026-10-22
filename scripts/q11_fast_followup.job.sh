#!/bin/bash
# GPU7: (1) TTFF 에서 T5 를 GPU 로 올렸을 때 (prepare 항 검증)  (2) int4 스케일 탐색으로 압축 화질 재측정 (캡 테스트 제외)
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD
R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-v2-14b-causal-fast
echo "########## LATENCY V2 T5-GPU ($(date)) ##########"
python -m specwm.latency_v2 --ckpt $CK --action-root $SCRATCH/rt2_action_05 --prompt-file $SCRATCH/p_05.txt --scene 05 --frames 161 \
  --arms fast_bf16,ours --areas 399360,184320 --t5-gpu --out $R/latency_v2_t5gpu
echo "########## EDGE SIM v3 int4 search ($(date)) ##########"
python -m specwm.edge_sim --ckpt $CK --image examples/05/image.jpg --action $SCRATCH/rt2_action_05 --prompt "$(cat $SCRATCH/p_05.txt)" \
  --frames 81 --local-attn 18 --kv-bits 4 --int4-search --no-cap --out $R/edge_sim_v3
touch $SCRATCH/exp/Q11_DONE; echo "[Q11_DONE] $(date)"
