#!/bin/bash
# ONE GPU-6 lock session (<= 15 min): P2 smoke clip on a Waypoint starter + constant-mouse sweep -> frame strips -> DA3 yaw recovery (K_yaw calibration)
# launched as:  flock /tmp/gpu6.lock env CUDA_VISIBLE_DEVICES=6 timeout 870 /home/hk28965/specwm/baselines/waypoint/smoke/run_smoke.sh
S=/home/hk28965/specwm; B=$S/baselines/waypoint; O=$B/smoke/wp__highland_castle_loch__P2; T0=$(date +%s)
echo "[$(date +%T)] lock acquired; GPU6 before: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i 6)"
timeout 600 $B/run.sh $S/baselines/scenes/wp/highland_castle_loch.png P2 1234 $O --sweep 0,60,120,240,480 --sweep-calls 30; echo "run.sh exit $? after $(( $(date +%s) - T0 )) s"
$B/frame_strip.sh $O/video.mp4 $O/frames
timeout 300 $S/scripts/envrun.sh python $B/calib_yaw.py $B/smoke/calib_da3.json P2_k118=$O/video.mp4:141.6 m0=$O/sweep_m0.mp4:0 m60=$O/sweep_m60.mp4:60 m120=$O/sweep_m120.mp4:120 m240=$O/sweep_m240.mp4:240 m480=$O/sweep_m480.mp4:480
echo "calib exit $?; [$(date +%T)] done after $(( $(date +%s) - T0 )) s; GPU6 after: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i 6)"
