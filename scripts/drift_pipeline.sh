#!/bin/bash
# wait for the drift dataset shards, then train the drift critic variants in parallel on GPUs 4-7
cd /home/hk28965/specwm; source venv/bin/activate; export SPECWM_NO_FLASH=1
until [ "$(grep -l DRIFT_DS_DONE exp/queue/drift_ds_{0,1,2,3}.log 2>/dev/null | wc -l)" -ge 4 ]; do sleep 60; done
echo "[drift] dataset ready: $(ls exp/drift_lat/*__s*.pt | wc -l) clips ($(date))"
C=lingbot-world-v2/specwm_out/drift; mkdir -p $C
CUDA_VISIBLE_DEVICES=4 python scripts/drift_train.py $C/main --label severe > exp/queue/drift_main.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python scripts/drift_train.py $C/noprev --label severe --no-prev > exp/queue/drift_noprev.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python scripts/drift_train.py $C/strict --label drift > exp/queue/drift_strict.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python scripts/drift_train.py $C/small --label severe --dim 128 --layers 4 > exp/queue/drift_small.log 2>&1 &
wait; grep -h "DRIFT_DONE" exp/queue/drift_main.log exp/queue/drift_noprev.log exp/queue/drift_strict.log exp/queue/drift_small.log | cut -c1-400
echo "[DRIFT_PIPE_DONE] $(date)"
