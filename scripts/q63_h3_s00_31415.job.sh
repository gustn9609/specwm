#!/bin/bash
# 서버는 이미 떠 있음 (GPU 0-3). S00 seed 31415 한 편만 생성 -> 공식 VR -> 서버 종료
true
S=$SCRATCH; cd $S; source $S/venv/bin/activate
python scripts/h3_gen.py --floor 00 --seeds 31415 --seconds 10
python scripts/h3_post.py
pkill -f "sglang serve" 2>/dev/null
R=$S/lingbot-world-v2/specwm_out; export HF_HOME=$S/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
$S/vr-venv/bin/python scripts/vr_official.py $R/vr/vr_official_floor.json h3_S00_s31415:$R/h3/floor_S00_s31415/video.mp4:$S/data/p_00.txt
touch $S/exp/Q63_DONE; echo "[Q63_DONE] $(date)"
