#!/bin/bash
# Exp 6: 360p(320x560) 2-step Fast draft 에 재접지 적용 — fast_alone / resync_k6(ideal, Base 480p→320x560 KV) / kvrw_laundered(360p target, r1_360) / kvrw_laundered480(480p target 다운샘플). 3 scenes x 3 seeds.
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V2=$S/lingbot-world-v2; O=$R/draft360_reground; mkdir -p $O; cd $V2; export PYTHONPATH=$V2
for SC in 05 01 02; do for SD in 1234 4321 9999; do
  echo "########## DRAFT360 REGROUND S$SC s$SD ($(date)) ##########"
  python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast --action-root $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --scene $SC --frames 161 --seeds $SD --base-video $R/floor/base/S${SC}_s$SD/video.mp4 --configs W18_sink6 --max-area 184320 --nodes 0,3 \
    --arms fast_alone,resync_k6,kvrw_laundered,kvrw_laundered480 --kv-providers "laundered=$R/r1_360/S$SC/target_s$SD/video.mp4,laundered480=$R/e1repro/S$SC/target_s$SD/video.mp4" --out $O/S$SC
done; done
touch $S/exp/Q119_DONE; echo "[Q119_DONE] $(date)"
