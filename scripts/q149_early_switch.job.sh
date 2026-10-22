#!/bin/bash
# Earlier correction on S05 seeds (480p): (a) switch at 3 s / 3.75 s with the existing target (kvrw_laundered_k4 / _k5); (b) proper early switch at 2.25 s: new target with RePaint prefix 3 latents (Base regenerates from 0.75 s) -> kvrw_early_k3.
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V1=$S/lingbot-world; V2=$S/lingbot-world-v2; CK1=$V1/lingbot-world-base-cam; SC=05; IMG=$V2/examples/$SC/image.jpg; O=$R/spec_dist/S$SC
for SD in $SEEDS; do BD=$R/floor/base/S${SC}_s$SD; TD=$O/target_s$SD; TE=$O/target_early_s$SD; LV=$(ls -d $O/launder_s$SD/REFINE_*/video.mp4 2>/dev/null | head -1)
  [ -f $TD/video.mp4 ] && [ -n "$LV" ] || { echo "[skip] $SD"; continue; }
  if [ ! -f $TE/video.mp4 ]; then echo "########## EARLY target (prefix 3) s$SD ($(date)) ##########"; cd $V1; PYTHONPATH=$V1 python -m specwm.bench_base_prefix2 --ckpt $CK1 --image $IMG --action-path $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --frames 121 --steps 16 --prefix-latents 3 --mode repaint --seed $SD --prefix-video $LV --out $TE 2>&1 | grep -E "^\[BASE|Error|Traceback"; fi
  echo "########## EARLY kvrw s$SD ($(date)) ##########"; cd $V2; PYTHONPATH=$V2 python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast --action-root $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --scene $SC --frames 161 --seeds $SD --base-video $BD/video.mp4 --configs W18_sink6 --arms kvrw_laundered_k4,kvrw_laundered_k5,kvrw_early_k3,kvrw_early_k4 --kv-providers "laundered=$TD/video.mp4,early=$TE/video.mp4" --out $O 2>&1 | grep -E "^\[W18|Error|Traceback"
done; touch $S/exp/Q149_${SHARD}_DONE; echo "[Q149_DONE $SHARD] $(date)"
