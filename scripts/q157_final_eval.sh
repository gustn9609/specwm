#!/bin/bash
cd /home/hk28965/specwm; until grep -q WORKER_DONE exp/logs/q173_w45.log 2>/dev/null; do sleep 300; done
while grep -q "bundle" exp/logs/q173_w67.log 2>/dev/null && [ "$(grep -c 'bundle .* done' exp/logs/q173_w67.log)" != "$(grep -c '=== \[6,7\] bundle' exp/logs/q173_w67.log | awk '{print $1/2}')" ]; do sleep 300; done
CUDA_VISIBLE_DEVICES=4 venv/bin/python scripts/spec_v0_eval.py > logs/spec_v0_eval_final.out 2>&1; touch exp/Q157_DONE; echo "[Q157_DONE] $(date)"
