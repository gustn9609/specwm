#!/bin/bash
# [10/20] device-tier PROFILE on an idle GPU (advisor's ask: fps / throughput / latency of the 24 GB device): solo arms rt2 (original: FP8 weights, bf16 KV, window 18),
# mx32 (FP8 KV, window 12, cap 32), mx24 (GPTQ-4 + FP8 KV, window 12, cap 24) on 6 clips of neutral_832 (832x480) and 6 of mg2 (640x352), RUNS_SUFFIX=_idle5.
# (recreated after the 10/20 23:15 reboot; run_ours.sh skips clips whose timing.json exists, so finished arms are no-ops)
G=5; B=/home/hk28965/specwm/baselines; SP=$(dirname "$0"); L=$SP/profile24.log
for SET in neutral_832 mg2; do for ARM in rt2 mx32 mx24; do
  echo "[$(date +%F_%H:%M)] $SET $ARM" | tee -a $L
  flock /tmp/gpu$G.lock env CUDA_VISIBLE_DEVICES=$G RUNS_SUFFIX=_idle5 $B/run_ours.sh $SET $ARM $G $SP/chunk_$SET.json >> $L 2>&1; echo "[$(date +%F_%H:%M)] $SET $ARM rc $?" | tee -a $L
done; done; echo PROFILE_ALL_DONE | tee -a $L
