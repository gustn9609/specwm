#!/bin/bash
# B-2 chain: wait for the 18 single-rater files and for the B-1 queue to finish, then encode the new clips, retrain the drift critic (big), compare.
cd /home/hk28965/specwm; source venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until [ "$(ls paper/overnight/blind_more/r1_p*.json 2>/dev/null | wc -l)" -ge 18 ] && [ "$(grep -l WORKER_DONE exp/queue/wcritv_{4,5,6,7}.log 2>/dev/null | wc -l)" -ge 4 ]; do sleep 60; done
echo "[drift2] labels + GPUs ready ($(date))"
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$((4+i)) python scripts/drift_dataset.py --shard $i/4 > exp/queue/drift_ds2_$i.log 2>&1 & done; wait
echo "[drift2] dataset: $(ls exp/drift_lat/*__s*.pt | wc -l) clips ($(date))"
C=lingbot-world-v2/specwm_out/drift
CUDA_VISIBLE_DEVICES=4 python scripts/drift_train.py $C/big_noprev --label severe --no-prev > exp/queue/drift_big_noprev.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python scripts/drift_train.py $C/big_main --label severe > exp/queue/drift_big_main.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python scripts/drift_train.py $C/big_noprev_noplanrc --label severe --no-prev --exclude-batch blind_planrc > exp/queue/drift_big_noprev_noplanrc.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python scripts/drift_train.py $C/big_strict --label drift --no-prev > exp/queue/drift_big_strict.log 2>&1 &
wait; grep -h "DRIFT_DONE" exp/queue/drift_big_*.log | cut -c1-300
CUDA_VISIBLE_DEVICES=4 python scripts/drift_plan_select.py $C/big_noprev_noplanrc/critic.pt exp/kveval/drift_plan_big_noplanrc.json > exp/queue/drift_plan_big_noplanrc.log 2>&1
grep -E "^AUC|^clean renders" exp/queue/drift_plan_big_noplanrc.log
echo "[DRIFT2_DONE] $(date)"
