#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

S=$SCRATCH
R=$S/lingbot-world-v2/specwm_out
V2=$S/lingbot-world-v2
CKF=$V2/lingbot-world-v2-14b-causal-fast
PR=$S/p_05.txt
H=$R/handoff

# E-A 4) 핸드오프 결과를 KV 제공자로 넣어 기기가 이어 그리게 하고, Base16 대비 회수율을 비교.
#        refine = 현행(다듬기 + RePaint, repro/S05/target_s4321), ideal = Base16 그 자체(상한선).
PROV="refine=$R/repro/S05/target_s4321/video.mp4,ideal=$R/repro/S05/base16_s4321/video.mp4"
for t in ho3 ho3_1step ho2; do
  [ -f $H/$t/corr/video.mp4 ] && PROV="$PROV,$t=$H/$t/corr/video.mp4" || echo "!!! provider $t missing, skipped"
done
ARMS="fast_alone,kvrw_refine,kvrw_ideal"
for t in ho3 ho3_1step ho2; do [ -f $H/$t/corr/video.mp4 ] && ARMS="$ARMS,kvrw_$t"; done
echo "##### E-A KV-rewrite comparison: arms=$ARMS #####"
cd $V2; PYTHONPATH=$V2 python -m specwm.resync_v4 --ckpt $CKF --action-root $S/rt2_action_05 --prompt-file $PR \
  --scene 05 --frames 161 --seeds 4321 --configs W18_sink6 --base16-video $R/repro/S05/base16_s4321/video.mp4 \
  --arms $ARMS --kv-providers "$PROV" --out $H/kvcmp || echo "!!! kvcmp FAILED"

echo "===== SUMMARY ====="
find $H/kvcmp -name "*.json" | head; ls $H/kvcmp/W18_sink6 2>/dev/null | tr '\n' ' '
echo "[R10KV_DONE]"
