#!/bin/bash
# Exp 4: 360p 2-step 의 축 분리 — StEvo 85 tasks: (a) 480p 2-step, (b) 360p 4-step. 이후 judge (GPU 동일) 로 채점.
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; cd $S/lingbot-world-v2; export PYTHONPATH=$PWD; CK=$PWD/lingbot-world-v2-14b-causal-fast
g() { [ $(ls $R/stevo/$1/*.mp4 2>/dev/null | wc -l) -ge 85 ] && return; echo "########## AXIS $1 ($(date)) ##########"; python -m specwm.stevo_gen --ckpt $CK --tasks $S/data/stevo/tasks_camera.json --out $R/stevo/$1 --arm fast_bf16 --frames 253 --t5-gpu "${@:2}"; }
g lingbot_fast_480_2step --max-area 399360 --nodes 0,3
g lingbot_fast_360_4step --max-area 184320
touch $S/exp/Q120_GEN_DONE
cd $S; while pgrep -f "sglang.launch_serve[r]" >/dev/null; do sleep 120; done
GPU=$CUDA_VISIBLE_DEVICES setsid nohup bash scripts/judge_serve.sh > logs/judge_serve_axis.out 2>&1 < /dev/null &
for i in $(seq 1 120); do curl -s --max-time 3 http://127.0.0.1:30100/health >/dev/null && break; sleep 10; done; sleep 20
for X in lingbot_fast_480_2step lingbot_fast_360_4step; do venv/bin/python scripts/stevo_prep_eval.py $R/stevo/$X; bash scripts/stevo_eval.sh $PWD/$R/stevo/$X $PWD/$R/stevo/runs > logs/stevo_eval_$X.out 2>&1; done
touch $S/exp/Q120_DONE; echo "[Q120_DONE] $(date)   (judge server left running for Exp 5 VQ)"
