#!/bin/bash
# GPU4 (VR 와 공유, 가벼움): CLIP 프롬프트 일치도 -> 모든 후속 큐(Q1,Q13,Q14) 종료 대기 -> 최종 분석/리포트/갤러리
true
source $SCRATCH/venv/bin/activate
echo "########## CLIP ADHERENCE ($(date)) ##########"
python $SCRATCH/scripts/clip_adherence.py
for i in $(seq 1 600); do [ -f $SCRATCH/exp/Q1_DONE ] && [ -f $SCRATCH/exp/Q13_DONE ] && [ -f $SCRATCH/exp/Q14_DONE ] && break; sleep 60; done
echo "########## FINAL ANALYSIS ($(date)) ##########"
python $SCRATCH/scripts/clip_adherence.py
python $SCRATCH/scripts/floor_analysis.py && python $SCRATCH/scripts/demo_gallery.py && python $SCRATCH/scripts/report_all.py
touch $SCRATCH/exp/Q15_DONE; echo "[Q15_DONE] $(date)"
