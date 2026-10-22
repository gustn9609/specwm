#!/bin/bash
# Verification session (<= 10 min, GPU lock): P2 on the same starter through the frozen single-clip run.sh path, K_yaw read from baselines/calib.json,
# then DA3 yaw recovery. GPU = CUDA_VISIBLE_DEVICES (10/17: GPU 4). Launch:
#   flock /tmp/gpu4.lock env CUDA_VISIBLE_DEVICES=4 timeout 600 /home/hk28965/specwm/baselines/waypoint/smoke/run_verify.sh [LABEL]   (LABEL default P2_k79)
S=/home/hk28965/specwm; B=$S/baselines/waypoint; LABEL=${1:-P2_k79}; O=$B/smoke/wp__highland_castle_loch__$LABEL; G=${CUDA_VISIBLE_DEVICES:-4}; T0=$(date +%s)
K=$($S/scripts/envrun.sh python -c "import json;print(json.load(open('$S/baselines/calib.json'))['waypoint']['K_yaw'])" 2>/dev/null | tail -1)
echo "[$(date +%T)] lock acquired; calib.json waypoint.K_yaw=$K; label $LABEL; GPU$G before: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i $G)"
timeout 480 $B/run.sh $S/baselines/scenes/wp/highland_castle_loch.png P2 1234 $O; echo "run.sh exit $? after $(( $(date +%s) - T0 )) s"
$B/frame_strip.sh $O/video.mp4 $O/frames
M=$($S/scripts/envrun.sh python -c "import json;print(json.load(open('$O/program.json'))['mouse_x_per_call_abs_max'])" 2>/dev/null | tail -1)
timeout 240 $S/scripts/envrun.sh python $B/calib_yaw.py $B/smoke/calib_da3.json $LABEL=$O/video.mp4:$M 2>&1 | grep -v "^\[INFO\|INFO \]"
echo "calib exit $?; [$(date +%T)] done after $(( $(date +%s) - T0 )) s; GPU$G after: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i $G)"
