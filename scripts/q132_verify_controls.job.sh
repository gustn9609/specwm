#!/bin/bash
# V-c: specificity controls for KV-rewrite on the 9 floor combos (480p, e1repro): provider = launder-only (no RePaint) / Base of ANOTHER seed / corrected target of ANOTHER seed / Fast's own video re-encoded (VAE round-trip sanity)
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V2=$S/lingbot-world-v2; O=$R/e1repro; cd $V2; export PYTHONPATH=$V2
nxt() { case $1 in 1234) echo 4321;; 4321) echo 9999;; 9999) echo 1234;; esac; }
for SC in 05 01 02; do for SD in 1234 4321 9999; do OS=$(nxt $SD)
  echo "########## CONTROLS S$SC s$SD (other seed $OS) ($(date)) ##########"
  python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast --action-root $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --scene $SC --frames 161 --seeds $SD --base-video $R/floor/base/S${SC}_s$SD/video.mp4 --configs W18_sink6 \
    --arms kvrw_launderonly,kvrw_baseother,kvrw_targetother,kvrw_fastself --kv-providers "launderonly=$O/S$SC/launder_s$SD/REFINE_0.40_0.20_0.08/video.mp4,baseother=$R/floor/base/S${SC}_s$OS/video.mp4,targetother=$O/S$SC/target_s$OS/video.mp4,fastself=$O/S$SC/W18_sink6/fast_alone_s$SD/video.mp4" --out $O/S$SC 2>&1 | grep -E "^\[|Error|Traceback"
done; done
touch $S/exp/Q132_DONE; echo "[Q132_DONE] $(date)"
