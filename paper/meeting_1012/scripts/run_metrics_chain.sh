#!/bin/bash
# [10/16] Speculative-Forcing-currency metrics on our clip sets, GPU 6 (shares the GPU with sf24 / the stage-4 evaluator: ~5 GB VBench, ~10 GB DA3).
# Two JSONs: paper/meeting_1012/data/vbench6.json (scripts/vbench6.py, tools/vbench_env) and data/rpe_da3.json (scripts/rpe_da3.py, main env).
# usage: ./run_metrics_chain.sh vbench | rpe
S=/home/hk28965/specwm; O=$S/lingbot-world-v2/specwm_out; TM=$O/twomach; D=$S/paper/meeting_1012/data; L=$S/paper/meeting_1012/logs; mkdir -p $D $L; cd $S
WHAT=$1
SETS=( "teacher704=$TM/a704_43/teacher/*/spec.mp4" "rt2=$TM/a704_43/rt2/*/spec.mp4" "adapt25=$TM/a704_43/adapt25/*/spec.mp4" "solo3=$TM/a704_43/solo3/*/spec.mp4"
       "adapt25c1hyb_dev=$TM/a704_43/adapt25c1hyb/device/*/spec.mp4" "adapt25c1hyb_cloud=$TM/a704_43/adapt25c1hyb/cloud/*/spec.mp4" "rt2c1hyb_cloud=$TM/a704_43/rt2c1hyb/cloud/*/spec.mp4"
       "a832_teacher=$TM/a832/teacher/*/spec.mp4" "a832_rt2=$TM/a832/rt2/*/spec.mp4" "a832_adapt25=$TM/a832/adapt25/*/spec.mp4" "a832_solo3=$TM/a832/solo3/*/spec.mp4"
       "a1280_teacher=$TM/a1280/teacher/*/spec.mp4" "a1280_rt2=$TM/a1280/rt2/*/spec.mp4" "a1280_solo3=$TM/a1280/solo3/*/spec.mp4" "abot1280=$O/abot/a1280/*/video.mp4" "abot704=$O/abot/a704/*/video.mp4" )
run() {   # run LABEL=GLOB ...
  if [ "$WHAT" = vbench ]; then TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=6 $S/tools/vbench_env/bin/python $S/scripts/vbench6.py $D/vbench6.json "$@" 2>&1 | grep "^\[vbench6\]\|n= \|Traceback\|Error" >> $L/vbench6.log
  else CUDA_VISIBLE_DEVICES=6 $S/scripts/envrun.sh python $S/scripts/rpe_da3.py $D/rpe_da3.json --stride 1 "$@" 2>&1 | grep "^\[rpe\]\|n= \|Traceback\|Error" >> $L/rpe_da3.log; fi
}
echo "[$(date +%H:%M)] chain $WHAT start" >> $L/$WHAT.chain.log
run "${SETS[@]}"
for arm in sf24 uniform4 uniform2; do   # 832 arms from the bench43_sf run, as they finish
  until grep -q "SF_DONE $arm" $O/bench43_sf/logs/$arm.out 2>/dev/null; do sleep 300; done
  run "sf_$arm=$O/bench43_sf/*/${arm}_s*/video.mp4"; echo "[$(date +%H:%M)] $arm done" >> $L/$WHAT.chain.log
done
echo "METRICS_CHAIN_DONE $WHAT $(date +%H:%M)" >> $L/$WHAT.chain.log
