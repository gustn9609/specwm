#!/bin/bash
# [10/16] their metrics on the 1.3B draft-ladder clips (4-step 832 w18 = the student arm; 4-step 704 w12 = fastest same-quality; 43 prompts seed 1234), GPU 6 small jobs
S=/home/hk28965/specwm; O=$S/lingbot-world-v2/specwm_out/draft_ladder; D=$S/paper/meeting_1012/data; L=$S/paper/meeting_1012/logs; cd $S
until grep -q "METRICS_CHAIN_DONE vbench" $L/vbench.chain.log 2>/dev/null; do sleep 300; done
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=6 $S/tools/vbench_env/bin/python $S/scripts/vbench6.py $D/vbench6.json "student1p3b_832=$O/s4_832_w18/*/spec.mp4" "student1p3b_704w12=$O/s4_704_w12/*/spec.mp4" 2>&1 | grep "^\[vbench6\]\|n= \|Traceback" >> $L/vbench6.log
CUDA_VISIBLE_DEVICES=6 $S/scripts/envrun.sh python $S/scripts/rpe_da3.py $D/rpe_da3.json --stride 1 "student1p3b_832=$O/s4_832_w18/*/spec.mp4" "student1p3b_704w12=$O/s4_704_w12/*/spec.mp4" 2>&1 | grep "^\[rpe\]\|n= \|Traceback" >> $L/rpe_da3.log
echo "M1P3B_DONE $(date +%H:%M)" >> $L/m1p3b.out
