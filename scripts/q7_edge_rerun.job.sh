#!/bin/bash
# Q3b 종료 후 GPU7: 캡을 정상상태(청크1 이후)에 거는 edge_sim 재실행 -> edge_sim_v2
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD
R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-v2-14b-causal-fast
for i in $(seq 1 300); do [ -f $SCRATCH/exp/Q3_DONE ] && break; sleep 60; done
echo "########## EDGE SIM v2 ($(date)) ##########"
python -m specwm.edge_sim --ckpt $CK --image examples/05/image.jpg --action $SCRATCH/rt2_action_05 --prompt "$(cat $SCRATCH/p_05.txt)" \
  --frames 81 --local-attn 18 --kv-bits 4 --caps 24,16,12 --out $R/edge_sim_v2
touch $SCRATCH/exp/Q7_DONE; echo "[Q7_DONE] $(date)"
