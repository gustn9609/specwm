#!/bin/bash
# SDVG + actions timing profile in the sdv2_latency format (scripts/wm_latency.py), on a quiet machine: waits for the bench-43 queue to finish,
# then runs the original stack (bf16 eager, as in the 10/5 slide) and the optimized stack (FP8 + Sage + compile + ring KV) on two GPUs.
# jar/castle x seeds 1234/4321/9999 (seed 1 = warm-up, skipped by wm_latency), verifiers: draft only / target only / SDVG tau -0.7 / -2.2
S=/home/hk28965/specwm; B=$S/lingbot-world-v2/specwm_out/bench43; O=specwm_out/sdvg_profile
until grep -q "queue empty" $B/queue2.log && [ $(ps aux | grep -c "[s]pecwm\.\(spec_ar_loop\|spec_early\|res_split_infer\)\|[r]un_mg2.py\|[r]un_waypoint_actions.py") -eq 0 ]; do sleep 60; done
echo "$(date) start profile"
for stack in 0 1; do
  G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v x=$stack '$1!=7 && $2<1000{n++; if(n==x+1){print $1; exit}}')
  NO_MP4=1 RESERVE_GB=85 CUDA_VISIBLE_DEVICES=$G bash $S/scripts/envrun.sh python -u -m specwm.spec_ar_loop --tasks sand_filling_up_jar,sandcastle_wave_impact --seeds 1,1234,4321,9999 \
     --verifiers none,target,ir~-0.7,ir~-2.2 --out $O/fast$stack --fast $stack --stream-decode 1 --save-lat 0 > $S/lingbot-world-v2/$O/fast$stack.log 2>&1 &
  sleep 5
done
wait; echo "$(date) PROFILE_DONE"
