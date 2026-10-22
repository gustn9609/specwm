#!/bin/bash
# [10/20 23:55, after the reboot] GPU 6 chain behind the soft-decode lock: (1) rs576r smoke (2 clips, neutral_832 balloon P1+P3) -> runs/ours_rs576r_neutral_832;
# (2) baseline-arm scorer loop (metrics_g7) on GPU 6 (was GPU 7 before 10/19; GPU 6 re-allowed 10/20 23:30).
B=/home/hk28965/specwm/baselines; C=$(dirname "$0"); L=$C/gpu6_chain.log; PK=$C/peak_rs576r.txt
echo "[$(date +%F_%H:%M)] rs576r smoke waiting for /tmp/gpu6.lock" >> $L
flock /tmp/gpu6.lock env CUDA_VISIBLE_DEVICES=6 $B/_gpu_peak_run.sh 6 $PK $B/gen/run_ressplit.sh neutral_832 rs576r 6 $C/smoke_rs576r.json >> $L 2>&1; rc=$?
echo "[$(date +%F_%H:%M)] rs576r smoke rc $rc peak(held_s MiB) $(cat $PK 2>/dev/null)" >> $L; echo "RS576R_SMOKE_DONE rc $rc" >> $L
echo "[$(date +%F_%H:%M)] scorer_split gpu6 metrics_g7 baselines" >> $L
exec $B/gen/scorer_split.sh 6 $B/metrics_g7 none waypoint mg2 abot astra astra_nocap abot_t5offload
