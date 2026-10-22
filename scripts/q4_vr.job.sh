#!/bin/bash
# Q4: MiniMax-H3 조사(CPU) -> VisionReward 데몬(생성되는 모든 영상 채점, Q1~Q3 끝날 때까지) -> DINO 플로어 분석 -> 종합 리포트. GPU 1장.
true
R=$SCRATCH/lingbot-world-v2/specwm_out
echo "########## H3 FIT PROBE ($(date)) ##########"
$SCRATCH/venv/bin/python $SCRATCH/scripts/h3_fit_probe.py || echo "[Q4] h3 probe 실패 (네트워크?)"
echo "########## VISIONREWARD DAEMON ($(date)) ##########"
for i in $(seq 1 120); do grep -q VR_SETUP_DONE $SCRATCH/logs/vr_setup.log 2>/dev/null && [ -x $SCRATCH/vr-venv/bin/python ] && break; sleep 60; done
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
$SCRATCH/vr-venv/bin/python $SCRATCH/scripts/vr_daemon.py --poll 180 --max-hours 40
echo "########## FLOOR ANALYSIS ($(date)) ##########"
$SCRATCH/venv/bin/python $SCRATCH/scripts/floor_analysis.py
echo "########## REPORT ($(date)) ##########"
$SCRATCH/venv/bin/python $SCRATCH/scripts/report_all.py
touch $SCRATCH/exp/Q4_DONE; echo "[Q4_DONE] $(date)"
