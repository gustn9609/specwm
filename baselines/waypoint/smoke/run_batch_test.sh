#!/bin/bash
# Batch-mode / K_yaw calibration session (<= 10 min, GPU lock): run.sh --batch on LIST.tsv (per-clip K_yaw in column 5), then frame strips, DA3 yaw
# recovery of every clip (labels b_<suffix> in smoke/calib_da3.json) and an optional frame diff of two clips (same seed -> determinism check).
#   flock /tmp/gpu4.lock env CUDA_VISIBLE_DEVICES=4 timeout 600 .../run_batch_test.sh TAG LIST.tsv [DIFF_A_DIR DIFF_B_DIR]
#   10/17 sessions: batch_k  smoke/batch_test.tsv (P2/P3 x K 79/64.7/53.6; diff vs the single-clip smoke/wp__highland_castle_loch__P2_k79)
#                   batch_k69 smoke/batch_k69.tsv (K 69: P2 s1234 x2 in one process, P2 s1235, P3; diff P2_k69 vs P2_k69_rep)
S=/home/hk28965/specwm; B=$S/baselines/waypoint; TAG=${1:-batch_k}; LIST=${2:-$B/smoke/batch_test.tsv}; O=$B/smoke/$TAG; G=${CUDA_VISIBLE_DEVICES:-4}; T0=$(date +%s)
echo "[$(date +%T)] lock acquired; tag $TAG list $LIST; GPU$G before: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i $G)"
timeout 420 $B/run.sh --batch $LIST $O; echo "run.sh --batch exit $? after $(( $(date +%s) - T0 )) s"
SPECS=""
for D in $O/wp__*; do
  [ -f $D/video.mp4 ] || continue; N=$(basename $D); N=${N#wp__highland_castle_loch__}
  $B/frame_strip.sh $D/video.mp4 $D/frames > /dev/null
  M=$($S/scripts/envrun.sh python -c "import json;print(json.load(open('$D/program.json'))['mouse_x_per_call_abs_max'])" 2>/dev/null | tail -1)
  SPECS="$SPECS b_$N=$D/video.mp4:$M"
done
timeout 300 $S/scripts/envrun.sh python $B/calib_yaw.py $B/smoke/calib_da3.json $SPECS 2>&1 | grep -v "^\[INFO\|INFO \]"
echo "calib exit $?"
[ -n "$4" ] && $S/scripts/envrun.sh python $B/frame_diff.py $3/video.mp4 $4/video.mp4 2>&1 | tail -3
echo "[$(date +%T)] done after $(( $(date +%s) - T0 )) s; GPU$G after: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i $G)"
