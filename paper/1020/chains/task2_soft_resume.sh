#!/bin/bash
# [10/20 23:50] resume of task2 after the reboot: inject_soft.py finished (ALL_DONE 22:32); redo the GPU decode step on GPU 6, then judge + CPU analysis.
S=/home/hk28965/specwm; E=$S/scripts/envrun.sh; L=/home/hk28965/specwm/paper/1020/chains/task2_soft.log
echo "[$(date +%F_%H:%M)] soft_decode on gpu6" >> $L
flock /tmp/gpu6.lock env CUDA_VISIBLE_DEVICES=6 $E python $S/paper/1020/soft_decode.py >> $L 2>&1; echo "[$(date +%F_%H:%M)] soft_decode rc $?" >> $L
echo TASK2_GPU_DONE >> $L
J=$S/lingbot-world-v2/specwm_out/inject/judge; JS=$S/lingbot-world-v2/specwm_out/inject/judge_soft
specs="T=$J/T_*_s*.mp4"; for a in alone soft@6_a0.25 soft@6_a0.5 soft@6_a0.75 once@6 prog@6; do specs="$specs $a=$JS/${a}_*_s*.mp4"; done
echo "[$(date +%F_%H:%M)] judge: $specs" >> $L
flock /tmp/vlm_base.lock env CUDA_VISIBLE_DEVICES="" $E python $S/scripts/vlm_clip_score.py $S/paper/1020/data/soft_vlm.json $specs >> $L 2>&1; echo "[$(date +%F_%H:%M)] judge rc $?" >> $L
CUDA_VISIBLE_DEVICES="" $E python $S/paper/1020/soft_analysis.py > /home/hk28965/specwm/paper/1020/chains/soft_analysis.out 2>&1; echo "[$(date +%F_%H:%M)] analysis rc $?" >> $L
echo TASK2_ALL_DONE >> $L
