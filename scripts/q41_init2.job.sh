#!/bin/bash
# GPU5 (Q38 후): oracle_init 노드2(σ=0.833) 변형 (이름 충돌로 스킵됐던 arm) 2시드 + geo
true
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
for i in $(seq 1 120); do [ -f $SCRATCH/exp/Q38_DONE ] && break; sleep 30; done
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-v2-14b-causal-fast
for SD in 1234 4321; do echo "########## PG oracle_init node2 s$SD ($(date)) ##########"
  python -m specwm.posegram --ckpt $CK --action-root $SCRATCH/traj55_action_05 --prompt-file $SCRATCH/p_05.txt --scene 05 --frames 873 --window 18 --sink 18 \
    --mode oracle_init --init-node 2 --seed $SD --name oracle_init_node2_s$SD --out $R/posegram/S05_W18 || echo "!!! FAILED"; done
python $SCRATCH/scripts/geo55.py $R/geo55.json --scan
touch $SCRATCH/exp/Q41_DONE; echo "[Q41_DONE] $(date)"
