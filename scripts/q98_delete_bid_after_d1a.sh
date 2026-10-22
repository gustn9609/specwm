#!/bin/bash
# D1a (bid launder) 가 끝나면 v2-bid 체크포인트(139 GB) 삭제. 재다운로드: huggingface-cli download robbyant/lingbot-world-v2-14b-bid
cd /home/hk28965/specwm; for i in $(seq 1 720); do [ -f exp/Q94_d1a_bidlaunder_DONE ] && break; sleep 60; done
ps -eo cmd | grep -q "[b]id_as_v1_cam" && { echo "bid still in use, abort"; exit 1; }
rm -rf lingbot-world-v2/lingbot-world-v2-14b-bid lingbot-world-v2/bid_as_v1_cam; echo "[BID_DELETED] $(date)"; df -h /home | tail -1
