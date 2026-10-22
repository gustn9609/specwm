#!/bin/bash
# clean, sequential timing pass on one otherwise idle GPU (after the SDVG profile): jar/castle x seeds 1234/4321/9999 (+ warm-up), 161 frames,
# streaming decode where the script supports it (per-frame ready times -> scripts/wm_latency.py). Output specwm_out/timing_pass/<arm>/...
S=/home/hk28965/specwm; O=specwm_out/timing_pass; L=$S/lingbot-world-v2/$O/logs; mkdir -p $L
until grep -q PROFILE_DONE $S/lingbot-world-v2/specwm_out/sdvg_profile/runner.log 2>/dev/null; do sleep 60; done
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1!=7 && $2<1000{print $1; exit}'); echo "$(date) timing pass on GPU $G"
T=sand_filling_up_jar,sandcastle_wave_impact; SD=1234,4321,9999
R="NO_MP4=1 SAVE_LAT=0 RESERVE_GB=85 CUDA_VISIBLE_DEVICES=$G bash $S/scripts/envrun.sh python -u -m specwm."
run(){ echo "$(date) $1"; eval "$2" > $L/$1.log 2>&1; }
run ours704 "${R}res_split_infer --tasks $T --seeds $SD --arm ours704 --out $O --fused --partial-commit 1 --lo-area 270336 --no-device --stream-decode 1 --no-video"
run ours832 "${R}res_split_infer --tasks $T --seeds $SD --arm ours832 --out $O --fused --partial-commit 1 --cloud-nodes 0,500 --lo-area 399360 --no-device --stream-decode 1 --no-video"
run c704_sep "${R}res_split_infer --tasks $T --seeds $SD --arm c704_sep --out $O --lo-area 270336 --no-device --stream-decode 1 --no-video"
run c704_fused "${R}res_split_infer --tasks $T --seeds $SD --arm c704_fused --out $O --fused --lo-area 270336 --no-device --stream-decode 1 --no-video"
run u2_832_sep "${R}res_split_infer --tasks $T --seeds $SD --arm u2_832_sep --out $O --cloud-nodes 0,500 --lo-area 399360 --no-device --stream-decode 1 --no-video"
run c576_pc1 "${R}res_split_infer --tasks $T --seeds $SD --arm c576_pc1 --out $O --fused --partial-commit 1 --lo-area 175104 --no-device --stream-decode 1 --no-video"
run teacher_se "${R}spec_early --mode uniform:4 --tasks $T --seeds $SD --out $O --no-video"
for cfg in "cw07_sep 0.7 separate" "cw07_pc1 0.7 partial1" "cw07_safe 0.7 safe" "cw05_pc1 0.5 partial1"; do set -- $cfg
  run $1 "${R}spec_early --mode diag:learned_dense --eps 0.24 --no-ref --chunk-frac $2 --commit $3 --arm $1 --tasks $T --seeds $SD --out $O --no-video"; done
python3 - <<'PY'
import json
S="/home/hk28965/specwm"; O=S+"/lingbot-world-v2/specwm_out"; T={t['id']:t for t in json.load(open(S+'/data/stevo/tasks_camera.json'))}
mj=[dict(image=T[t]['image'], cond=O+"/mg2/bench_cond/%s.json"%t, out=O+"/timing_pass/%s/mg2_s%d"%(t,s), seed=s) for t in ("sand_filling_up_jar","sandcastle_wave_impact") for s in (1234,4321,9999)]
wj=[dict(image=T[t]['image'], actions=O+"/waypoint/bench_act/%s.json"%t, out=O+"/timing_pass/%s/wp_s%d"%(t,s), seed=s) for t in ("sand_filling_up_jar","sandcastle_wave_impact") for s in (1234,4321,9999)]
json.dump(mj, open(O+"/timing_pass/mg2_jobs.json","w")); json.dump(wj, open(O+"/timing_pass/wp_jobs.json","w"))
PY
run mg2 "cd $S && PYTHONPATH=$S/tools/mg2/shim CPATH=$S/tools/pydev/usr/include/python3.12:$S/tools/pydev/usr/include/x86_64-linux-gnu/python3.12:$S/tools/pydev/usr/include OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=$G venv/bin/python tools/mg2/run_mg2.py --jobs $S/lingbot-world-v2/$O/mg2_jobs.json"
run wp "cd $S && OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=$G venv/bin/python tools/waypoint/run_waypoint_actions.py --warm 8 --compile --jobs $S/lingbot-world-v2/$O/wp_jobs.json"
echo "$(date) TIMING_DONE"
