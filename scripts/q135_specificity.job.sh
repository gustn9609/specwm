#!/bin/bash
# P3 specificity: RePaint conditioned more strongly on the Fast prefix — prefix 15 latents; prefix 9 + SDEdit start sigma 0.8 / 0.6 (init from laundered video, 121 f) -> kvrw arms. 5 combos.
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V1=$S/lingbot-world; V2=$S/lingbot-world-v2; CK=$V1/lingbot-world-base-cam; O=$R/e1repro
for CS in "05 1234" "05 4321" "05 9999" "01 1234" "02 1234"; do set -- $CS; SC=$1; SD=$2; IMG=$V2/examples/$SC/image.jpg; LV=$(ls -d $O/S$SC/launder_s$SD/REFINE_*/video.mp4 | head -1); PROV=""
  for VAR in "p15 --prefix-latents 15" "sd08 --prefix-latents 9 --sdedit-sigma 0.8" "sd06 --prefix-latents 9 --sdedit-sigma 0.6" "base121 --prefix-latents 9"; do set -- $VAR; NAME=$1; shift; TD=$O/S$SC/target_${NAME}_s$SD
    if [ ! -f $TD/video.mp4 ]; then echo "########## SPEC $NAME S$SC s$SD ($(date)) ##########"; cd $V1; PYTHONPATH=$V1 python -m specwm.bench_base_prefix2 --ckpt $CK --image $IMG --action-path $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --frames 121 --steps 16 --mode repaint --seed $SD --prefix-video $LV --out $TD "$@" 2>&1 | grep -E "^\[|Error|Traceback"; fi
    PROV="$PROV,$NAME=$TD/video.mp4"
  done
  echo "########## SPEC KVRW S$SC s$SD ($(date)) ##########"; cd $V2; PYTHONPATH=$V2 python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast --action-root $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --scene $SC --frames 161 --seeds $SD --base-video $R/floor/base/S${SC}_s$SD/video.mp4 --configs W18_sink6 --arms kvrw_p15,kvrw_sd08,kvrw_sd06,kvrw_base121 --kv-providers "${PROV#,}" --out $O/S$SC 2>&1 | grep -E "^\[|Error|Traceback"
done
touch $S/exp/Q135_DONE; echo "[Q135_DONE] $(date)"
