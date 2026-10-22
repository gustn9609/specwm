#!/bin/bash
# [10/20 23:55] waits for the device-tier profile (PROFILE_ALL_DONE in the profile log) so the profile keeps an idle GPU 5, then runs the ours_* scorer loop (metrics/) on GPU 5.
B=/home/hk28965/specwm/baselines; C=$(dirname "$0"); PL=/tmp/claude-1741758895/-home-hk28965/76e7a8ef-ae2e-455b-8c73-052f5bde2a30/scratchpad/profile24/profile24.log
while ! grep -q PROFILE_ALL_DONE $PL 2>/dev/null; do sleep 120; done
echo "[$(date +%F_%H:%M)] profile done -> scorer_split gpu5 metrics ours_*" >> $C/gpu5_after_profile.log
exec $B/gen/scorer_split.sh 5 $B/metrics none "ours_*"
