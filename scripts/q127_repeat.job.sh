#!/bin/bash
# Item 1: 분산 원인 — 같은 시드로 3회 반복 (S05 s1234, S02 s9999; arms fast_alone + kvrw_laundered, E1' 480p target 재사용). run-to-run 차이 vs seed/scene 차이.
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V2=$S/lingbot-world-v2; cd $V2; export PYTHONPATH=$V2
for REP in 1 2 3; do for CS in "05 1234" "02 9999"; do set -- $CS; SC=$1; SD=$2; O=$R/repeat/rep$REP
  echo "########## REPEAT rep$REP S$SC s$SD ($(date)) ##########"
  python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast --action-root $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --scene $SC --frames 161 --seeds $SD --base-video $R/floor/base/S${SC}_s$SD/video.mp4 --configs W18_sink6 \
    --arms fast_alone,resync_k6,kvrw_laundered --kv-providers "laundered=$R/e1repro/S$SC/target_s$SD/video.mp4" --out $O/S$SC 2>&1 | grep -E "^\[|Error|Traceback"
done; done
touch $S/exp/Q127_DONE; echo "[Q127_DONE] $(date)"
