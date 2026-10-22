#!/bin/bash
# GPU4 (Q21 후): (1) S01 아노말리 점검 - 플로어 영상(Base 6시드/Fast 5시드 x 4씬) 에 geo probe (2) Base 16-step 55s S05 (~2h) (3) geo 재실행
true
for i in $(seq 1 240); do [ -f $SCRATCH/exp/Q21_DONE ] && break; sleep 60; done
source $SCRATCH/venv/bin/activate; R=$SCRATCH/lingbot-world-v2/specwm_out
echo "########## GEO floor (S01 anomaly) ($(date)) ##########"
python $SCRATCH/scripts/geo55.py $R/geo55.json --scan
echo "########## BASE16 55s S05 ($(date)) ##########"
SC=05 STEPS=16 bash $SCRATCH/scripts/q30_base55.job.sh
python $SCRATCH/scripts/geo55.py $R/geo55.json --scan
touch $SCRATCH/exp/Q33_DONE; echo "[Q33_DONE] $(date)"
