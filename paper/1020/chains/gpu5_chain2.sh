#!/bin/bash
# [10/20 23:58] GPU 5 chain after the device-tier profile's first pass: (1) GPTQ-4 save (tier-8 chunk 310, copied with the GPU guard widened; the queue's
# original is parked in claimed/310_gptq4_save.sh.gpu5 and moved to done/ on success, back to 8/ on failure); (2) re-run profile24.sh so the mx24 arms
# (which need the GPTQ-4 file) get measured on the idle GPU; (3) the ours_* scorer loop (metrics/) on GPU 5.
B=/home/hk28965/specwm/baselines; C=/home/hk28965/specwm/paper/1020/chains; Q=/home/hk28965/specwm/baselines/gen/queue; PL=/tmp/claude-1741758895/-home-hk28965/76e7a8ef-ae2e-455b-8c73-052f5bde2a30/scratchpad/profile24/profile24.log; L=$C/gpu5_chain2.log
while ! grep -q PROFILE_ALL_DONE $PL 2>/dev/null; do sleep 120; done
echo "[$(date +%F_%H:%M)] profile pass 1 done -> gptq4 save on gpu5" >> $L
bash $C/310_gptq4_save_gpu5.sh 5 >> $L 2>&1; rc=$?; echo "[$(date +%F_%H:%M)] gptq4 save rc $rc" >> $L
if [ $rc -eq 0 ] && [ -f $B/gen/sentinels/gptq4_save.ok ]; then mv $Q/claimed/310_gptq4_save.sh.gpu5 $Q/done/310_gptq4_save.sh.gpu5; echo GPTQ4_SAVE_OK >> $L
  bash $SP/profile24/profile24.sh >> $L 2>&1; echo "[$(date +%F_%H:%M)] profile pass 2 rc $? ; PROFILE_MX24_DONE" >> $L
else mv $Q/claimed/310_gptq4_save.sh.gpu5 $Q/8/310_gptq4_save.sh; echo GPTQ4_SAVE_FAIL >> $L; fi
echo "[$(date +%F_%H:%M)] scorer_split gpu5 metrics ours_*" >> $L
exec $B/gen/scorer_split.sh 5 $B/metrics none "ours_*"
