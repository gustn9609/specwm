#!/bin/bash
# P2 seam removal: RePaint variants (121 f, 480p) on 5 combos -> kvrw arms -> (later) DINO + VGGT seam at 2.25 s. Variants: u2 (resample 2), blend6 (6 latents soft seam), blend6_u2. Reuses E1' laundered videos.
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V1=$S/lingbot-world; V2=$S/lingbot-world-v2; CK=$V1/lingbot-world-base-cam; O=$R/e1repro
for CS in "05 1234" "05 4321" "05 9999" "01 1234" "02 1234"; do set -- $CS; SC=$1; SD=$2; IMG=$V2/examples/$SC/image.jpg; LV=$(ls -d $O/S$SC/launder_s$SD/REFINE_*/video.mp4 | head -1); PROV=""
  for VAR in "u2 --resample 2" "blend6 --blend-latents 6" "blend6_u2 --blend-latents 6 --resample 2"; do set -- $VAR; NAME=$1; shift; TD=$O/S$SC/target_${NAME}_s$SD
    if [ ! -f $TD/video.mp4 ]; then echo "########## SEAM $NAME S$SC s$SD ($(date)) ##########"; cd $V1; PYTHONPATH=$V1 python -m specwm.bench_base_prefix2 --ckpt $CK --image $IMG --action-path $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --frames 121 --steps 16 --prefix-latents 9 --mode repaint --seed $SD --prefix-video $LV --out $TD "$@" 2>&1 | grep -E "^\[|Error|Traceback"; fi
    PROV="$PROV,$NAME=$TD/video.mp4"
  done
  echo "########## SEAM KVRW S$SC s$SD ($(date)) ##########"; cd $V2; PYTHONPATH=$V2 python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast --action-root $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --scene $SC --frames 161 --seeds $SD --base-video $R/floor/base/S${SC}_s$SD/video.mp4 --configs W18_sink6 --arms kvrw_u2,kvrw_blend6,kvrw_blend6_u2 --kv-providers "${PROV#,}" --out $O/S$SC 2>&1 | grep -E "^\[|Error|Traceback"
done
touch $S/exp/Q134_DONE; echo "[Q134_DONE] $(date)"
