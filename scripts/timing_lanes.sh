#!/bin/bash
# remaining timing-pass arms in 2 lanes (lane = GPU; arms sequential within a lane); seeds 1 (warm-up per task, excluded) + 1234/4321/9999
S=/home/hk28965/specwm; O=specwm_out/timing_pass; L=$S/lingbot-world-v2/$O/logs; G=$1; shift
T=sand_filling_up_jar,sandcastle_wave_impact; SD=1,1234,4321,9999
R="NO_MP4=1 SAVE_LAT=0 RESERVE_GB=85 CUDA_VISIBLE_DEVICES=$G bash $S/scripts/envrun.sh python -u -m specwm."
while [ $(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits) -gt 1000 ]; do sleep 20; done
for arm in "$@"; do echo "$(date) $arm on GPU $G"
  case $arm in
    c704_fused) C="${R}res_split_infer --tasks $T --seeds $SD --arm c704_fused --out $O --fused --lo-area 270336 --no-device --stream-decode 1 --no-video --warm 0";;
    u2_832_sep) C="${R}res_split_infer --tasks $T --seeds $SD --arm u2_832_sep --out $O --cloud-nodes 0,500 --lo-area 399360 --no-device --stream-decode 1 --no-video --warm 0";;
    c576_pc1) C="${R}res_split_infer --tasks $T --seeds $SD --arm c576_pc1 --out $O --fused --partial-commit 1 --lo-area 175104 --no-device --stream-decode 1 --no-video --warm 0";;
    teacher_se) C="${R}spec_early --mode uniform:4 --tasks $T --seeds $SD --out $O --no-video --warm 0";;
    cw07_sep) C="${R}spec_early --mode diag:learned_dense --eps 0.24 --no-ref --chunk-frac 0.7 --commit separate --arm cw07_sep --tasks $T --seeds $SD --out $O --no-video --warm 0";;
    cw07_pc1) C="${R}spec_early --mode diag:learned_dense --eps 0.24 --no-ref --chunk-frac 0.7 --commit partial1 --arm cw07_pc1 --tasks $T --seeds $SD --out $O --no-video --warm 0";;
    cw07_safe) C="${R}spec_early --mode diag:learned_dense --eps 0.24 --no-ref --chunk-frac 0.7 --commit safe --arm cw07_safe --tasks $T --seeds $SD --out $O --no-video --warm 0";;
    cw05_pc1) C="${R}spec_early --mode diag:learned_dense --eps 0.24 --no-ref --chunk-frac 0.5 --commit partial1 --arm cw05_pc1 --tasks $T --seeds $SD --out $O --no-video --warm 0";;
    mg2) C="cd $S && PYTHONPATH=$S/tools/mg2/shim CPATH=$S/tools/pydev/usr/include/python3.12:$S/tools/pydev/usr/include/x86_64-linux-gnu/python3.12:$S/tools/pydev/usr/include OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=$G venv/bin/python tools/mg2/run_mg2.py --jobs $S/lingbot-world-v2/$O/mg2_jobs.json";;
    wp) C="cd $S && OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=$G venv/bin/python tools/waypoint/run_waypoint_actions.py --warm 8 --compile --jobs $S/lingbot-world-v2/$O/wp_jobs.json";;
  esac
  eval "$C" > $L/$arm.log 2>&1
done
echo "$(date) LANE_${G}_DONE"
