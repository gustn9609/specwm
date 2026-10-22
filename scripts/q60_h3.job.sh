#!/bin/bash
# MiniMax-H3 체인 (GPU 0-3): 설치·다운로드 대기 -> sglang 서버 -> (A) 플로어: 씬 5 x 시드 3 (B) 데모 프롬프트 16 -> 후처리 -> VR/CLIP 채점
true
S=$SCRATCH; cd $S
for i in $(seq 1 240); do grep -q H3_SETUP_DONE logs/h3_setup.log 2>/dev/null && grep -qE "Fetching 82 files: 100%|100%\|" logs/h3_download.log 2>/dev/null && ! pgrep -f "hf download MiniMaxAI" >/dev/null && break; sleep 30; done
grep -q H3_SETUP_DONE logs/h3_setup.log || { echo "[Q60] sglang 설치 실패/미완"; tail -20 logs/h3_setup.log; exit 1; }
echo "########## H3 SERVE ($(date)) ##########"
nohup bash scripts/h3_serve.sh > logs/h3_serve.out 2>&1 &
SP=$!
for i in $(seq 1 120); do curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:30010/health 2>/dev/null | grep -qE "200" && break; grep -qiE "Traceback|error" logs/h3_serve.out && { echo "[Q60] 서버 오류"; grep -iE "Error|Traceback" -A3 logs/h3_serve.out | tail -20; }; sleep 30; done
source $S/venv/bin/activate
echo "########## H3 GEN floor ($(date)) ##########"
python scripts/h3_gen.py --floor 05,01,00,02,03 --seeds 1234,4321,9999 --seconds 10
echo "########## H3 GEN demo ($(date)) ##########"
python scripts/h3_gen.py --demo $S/exp/demo_list.json,$S/exp/demo_list2.json --seconds 10
python scripts/h3_post.py
kill $SP 2>/dev/null; sleep 5; pkill -f "sglang serve" 2>/dev/null
echo "########## H3 VR/CLIP ($(date)) ##########"
export HF_HOME=$S/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; export CUDA_VISIBLE_DEVICES=4
$S/vr-venv/bin/python scripts/vr_daemon.py --poll 30 --max-hours 2 --roots h3
touch $S/exp/Q60_DONE; echo "[Q60_DONE] $(date)"
