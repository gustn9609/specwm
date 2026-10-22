#!/bin/bash
# GPU 0-3: H3 플로어 시드 보강 (2718, 31415 x 4씬 = 8편) -> Base/Fast 와 같은 5시드 -> 공식 VR 채점
true
S=$SCRATCH; cd $S; export PATH=$HOME/.local/bin:$PATH
echo "########## H3 SERVE ($(date)) ##########"; nohup bash scripts/h3_serve.sh > logs/h3_serve2.out 2>&1 & SP=$!
for i in $(seq 1 60); do curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:30010/health 2>/dev/null | grep -q 200 && break; sleep 20; done
source $S/venv/bin/activate
python scripts/h3_gen.py --floor 05,01,00,02 --seeds 2718,31415 --seconds 10
python scripts/h3_post.py
kill $SP 2>/dev/null; sleep 5; pkill -f "sglang serve" 2>/dev/null
R=$S/lingbot-world-v2/specwm_out; ARGS=""; for sc in 05 01 00 02; do for s in 2718 31415; do ARGS="$ARGS h3_S${sc}_s$s:$R/h3/floor_S${sc}_s$s/video.mp4:$S/data/p_$sc.txt"; done; done
export HF_HOME=$S/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
$S/vr-venv/bin/python scripts/vr_official.py $R/vr/vr_official_floor.json $ARGS
touch $S/exp/Q62_DONE; echo "[Q62_DONE] $(date)"
