#!/bin/bash
# 사용: ./run.sh <gpu번호|auto> <스크립트.sh> [태그]   -> 백그라운드, 로그 $SCRATCH/logs/<태그>.<pid>.out
source /home/hk28965/specwm/env.sh; mkdir -p $SCRATCH/logs
G=$1; S=$2; T=${3:-$(basename $S .sh)}
if [ "$G" = auto ]; then G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F", " "\$2 < 2000 {print \$1; exit}"); [ -z "$G" ] && { echo "빈 GPU 없음"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; exit 1; }; fi
CUDA_VISIBLE_DEVICES=$G nohup bash "$S" > $SCRATCH/logs/$T.$$.out 2>&1 &
echo "gpu$G pid $! log $SCRATCH/logs/$T.$$.out"
