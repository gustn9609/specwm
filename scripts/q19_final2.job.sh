#!/bin/bash
# GPU4: 모든 후속(Q1,Q13,Q14,Q16,Q17) 종료 대기 -> VR 잔여 채점 -> 최종 분석/갤러리/리포트
true
source $SCRATCH/venv/bin/activate
for i in $(seq 1 720); do [ -f $SCRATCH/exp/Q1_DONE ] && [ -f $SCRATCH/exp/Q13_DONE ] && [ -f $SCRATCH/exp/Q14_DONE ] && [ -f $SCRATCH/exp/Q16_DONE ] && [ -f $SCRATCH/exp/Q17_DONE ] && break; sleep 60; done
echo "########## VR catch-up ($(date)) ##########"
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
$SCRATCH/vr-venv/bin/python $SCRATCH/scripts/vr_daemon.py --poll 60 --max-hours 2 --roots floor,handoff_grid,ctx_grid,revisit,repro,demo,handoff
echo "########## FINAL2 ANALYSIS ($(date)) ##########"
python $SCRATCH/scripts/clip_adherence.py; python $SCRATCH/scripts/floor_analysis.py && python $SCRATCH/scripts/demo_gallery.py && python $SCRATCH/scripts/report_all.py
touch $SCRATCH/exp/Q19_DONE; echo "[Q19_DONE] $(date)"
