#!/bin/bash
# run queued fused_chain jobs ("SEEDS ARM ARGS...") one per free GPU as GPUs free up
S=/home/hk28965/specwm; Q=$1
while [ -s $Q ]; do
  G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<1000{print $1; exit}')
  [ -n "$G" ] || { sleep 5; continue; }
  L=$(head -1 $Q); sed -i 1d $Q; set -- $L; SD=$1; shift
  echo "$(date) GPU $G: $L"; RESERVE_GB=85 SEEDS=$SD nohup bash $S/scripts/fused_chain.sh $G "$@" >/dev/null 2>&1 &
  sleep 60
done
