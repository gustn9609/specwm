#!/bin/bash
# GPU5 E0b: 강한 oracle (정확 pose 탐색) — KV 주입 / σ1 초기화 / 전 노드 블렌딩, 2시드 -> geo scan
true
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
for i in $(seq 1 120); do [ -f $SCRATCH/exp/PG_SMOKE_OK ] && break; sleep 30; done
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-v2-14b-causal-fast; O=$R/posegram/S05_W18
run() { echo "########## PG $* ($(date)) ##########"; python -m specwm.posegram --ckpt $CK --action-root $SCRATCH/traj55_action_05 --prompt-file $SCRATCH/p_05.txt --scene 05 --frames 873 --window 18 --sink 18 --out $O "$@" || echo "!!! FAILED $*"; }
for SD in 1234 4321; do
  run --mode ctrl --seed $SD
  run --mode oracle_kv --seed $SD
  run --mode oracle_init --init-node 1 --seed $SD
  run --mode oracle_blend_all --alpha 0.3 --seed $SD
  run --mode oracle_blend_all --alpha 0.5 --seed $SD
done
run --mode oracle_init --init-node 2 --seed 1234
python $SCRATCH/scripts/geo55.py $R/geo55.json --scan
touch $SCRATCH/exp/Q38_DONE; echo "[Q38_DONE] $(date)"
