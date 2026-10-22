#!/bin/bash
# Q3b: Q3 후속 (Base S00/S02 참조 대기로 GPU 가 노는 것을 피해 순서 변경): latency_v2 -> profile -> edge_sim -> ctx_grid S05 다중 시드 -> ctx S00/S02 (참조 있으면)
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD
R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-v2-14b-causal-fast
echo "########## LATENCY V2 ($(date)) ##########"
python -m specwm.latency_v2 --ckpt $CK --action-root $SCRATCH/rt2_action_05 --prompt-file $SCRATCH/p_05.txt --scene 05 --frames 161 \
  --arms fast_bf16,fast_kv4,ours --areas 399360,184320 --out $R/latency_v2
echo "########## PROFILE ($(date)) ##########"
python -m specwm.profile_phases --ckpt $CK --image examples/05/image.jpg --action $SCRATCH/rt2_action_05 --prompt "$(cat $SCRATCH/p_05.txt)" \
  --frames 81 --local-attn 18 --warmup-chunks 2 --steady-from 6 --hook-blocks 3 --kernel-trace --out $R/profile
echo "########## EDGE SIM ($(date)) ##########"
python -m specwm.edge_sim --ckpt $CK --image examples/05/image.jpg --action $SCRATCH/rt2_action_05 --prompt "$(cat $SCRATCH/p_05.txt)" \
  --frames 81 --local-attn 18 --kv-bits 4 --caps 24,16,12 --out $R/edge_sim
touch $SCRATCH/exp/Q3_CORE_DONE
echo "########## CTX GRID S05 seeds 4321,9999 ($(date)) ##########"
python -m specwm.ctx_grid --ckpt $CK --action-root $SCRATCH/rt2_action_05 --prompt-file $SCRATCH/p_05.txt --scene 05 --frames 161 \
  --base-video $R/floor/base/S05_s1234/video.mp4 --res 1.0,0.75,0.5 --bits 16,8,4 --sigmas 0,0.625,0.833 --seeds 4321,9999 --out $R/ctx_grid/S05
for SC in 00 02; do
  BV=$R/floor/base/S${SC}_s1234/video.mp4; [ -f $BV ] || BV=$R/floor/base/S${SC}_s4321/video.mp4
  [ -f $BV ] || { echo "[Q3b] S$SC base 참조 없음, ctx_grid 건너뜀"; continue; }
  echo "########## CTX GRID S$SC ($(date)) ##########"
  python -m specwm.ctx_grid --ckpt $CK --action-root $SCRATCH/rt2_action_$SC --prompt-file $SCRATCH/p_$SC.txt --scene $SC --frames 161 \
    --base-video $BV --res 1.0,0.75,0.5 --bits 16,8,4 --sigmas 0,0.625,0.833 --seeds 1234 --out $R/ctx_grid/S$SC
done
touch $SCRATCH/exp/Q3_DONE; echo "[Q3_DONE] $(date)"
