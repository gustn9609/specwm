#!/bin/bash
# Q3: Fast 큐 - 플로어(5시드x4씬) -> 다운링크 컨텍스트 그리드 -> TTFF/steady 지연 -> 프로파일 -> 엣지 시뮬. GPU 1장.
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD
R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-v2-14b-causal-fast
# 1) Fast 노이즈 플로어
for SC in 05 01 00 02; do
  echo "########## FAST floor S$SC ($(date)) ##########"
  python -m specwm.resync_v4 --ckpt $CK --action-root $SCRATCH/rt2_action_$SC --prompt-file $SCRATCH/p_$SC.txt --scene $SC --frames 161 \
    --arms fast_alone --configs W18_sink6 --seeds 1234,4321,9999,2718,31415 --out $R/floor/fast/S$SC
done
# 2) 다운링크 컨텍스트 그리드 (Base 참조 영상 필요: S05/S01 은 있음, S00/S02 는 Q1 이 만들면)
for SC in 05 01 00 02; do
  BV=$R/floor/base/S${SC}_s1234/video.mp4; [ -f $BV ] || BV=$R/floor/base/S${SC}_s4321/video.mp4
  if [ ! -f $BV ]; then for i in $(seq 1 180); do BV=$R/floor/base/S${SC}_s4321/video.mp4; [ -f $BV ] && break; sleep 60; done; fi
  [ -f $BV ] || { echo "[Q3] S$SC base video 없음, ctx_grid 건너뜀"; continue; }
  echo "########## CTX GRID S$SC ($(date)) base=$BV ##########"
  python -m specwm.ctx_grid --ckpt $CK --action-root $SCRATCH/rt2_action_$SC --prompt-file $SCRATCH/p_$SC.txt --scene $SC --frames 161 \
    --base-video $BV --res 1.0,0.75,0.5 --bits 16,8,4 --sigmas 0,0.625,0.833 --seeds 1234 --out $R/ctx_grid/S$SC
done
# 3) TTFF vs steady
echo "########## LATENCY V2 ($(date)) ##########"
python -m specwm.latency_v2 --ckpt $CK --action-root $SCRATCH/rt2_action_05 --prompt-file $SCRATCH/p_05.txt --scene 05 --frames 161 \
  --arms fast_bf16,fast_kv4,ours --areas 399360,184320 --out $R/latency_v2
# 4) 프로파일 (커널 트레이스 포함)
echo "########## PROFILE ($(date)) ##########"
python -m specwm.profile_phases --ckpt $CK --image examples/05/image.jpg --action $SCRATCH/rt2_action_05 --prompt "$(cat $SCRATCH/p_05.txt)" \
  --frames 81 --local-attn 18 --warmup-chunks 2 --steady-from 6 --hook-blocks 3 --kernel-trace --out $R/profile
# 5) 엣지 시뮬 (메모리 캡 + roofline 투영)
echo "########## EDGE SIM ($(date)) ##########"
python -m specwm.edge_sim --ckpt $CK --image examples/05/image.jpg --action $SCRATCH/rt2_action_05 --prompt "$(cat $SCRATCH/p_05.txt)" \
  --frames 81 --local-attn 18 --kv-bits 4 --caps 24,16,12 --out $R/edge_sim
touch $SCRATCH/exp/Q3_DONE; echo "[Q3_DONE] $(date)"
