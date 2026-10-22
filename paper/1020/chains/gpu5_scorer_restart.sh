#!/bin/bash
# 10/21 08:55: wait for the running score_runs (ours_rt2c1hyb_dev_neutral_832, pid $1) to finish, then restart the GPU-5 loop on the arms not given to GPUs 0-3
B=/home/hk28965/specwm/baselines; while kill -0 $1 2>/dev/null; do sleep 30; done
exec bash $B/gen/scorer_split.sh 5 $B/metrics none 'ours_student1p3b_*' 'ours_rt2c1hyb_dev_neutral_832' 'ours_adapt25*' 'ours_rt2_*' 'ours_teacher_*' 'ours_rt2c1hyb_*' 'ours_sf24_*'
