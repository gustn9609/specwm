#!/bin/bash
# Periodic correction on S05 seeds: (a) ideal periodic = Base ctx re-injected every 2/3/4 chunks (every2/every3/every4); (b) ours x2 = second KV-rewrite at chunk 10 built on the once-corrected rollout
# (launder its latent -> RePaint prefix 21 latents (81 f) -> kvrw_<second>_k10). SEEDS per shard.
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V1=$S/lingbot-world; V2=$S/lingbot-world-v2; CK1=$V1/lingbot-world-base-cam; SC=05; IMG=$V2/examples/$SC/image.jpg; O=$R/spec_dist/S$SC
for SD in $SEEDS; do BD=$R/floor/base/S${SC}_s$SD; [ -f $O/W18_sink6/kvrw_laundered_s$SD/latent.pt ] || { echo "[skip] $SD"; continue; }
  echo "########## PERIODIC ideal s$SD ($(date)) ##########"; cd $V2; PYTHONPATH=$V2 python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast --action-root $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --scene $SC --frames 161 --seeds $SD --base-video $BD/video.mp4 --configs W18_sink6 --arms every2,every3,every4 --out $O 2>&1 | grep -E "^\[|Error|Traceback"
  LD=$O/launder2_s$SD; TD=$O/target2_s$SD
  if [ ! -f $TD/video.mp4 ]; then echo "########## PERIODIC ours-2nd launder s$SD ($(date)) ##########"; cd $V1; PYTHONPATH=$V1 python -m specwm.hybrid_refine --ckpt $CK1 --scene $SC --image $IMG --action-dir $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --fast-latent $O/W18_sink6/kvrw_laundered_s$SD/latent.pt --frames 153 --schedules "0.40,0.20,0.08" --out $LD 2>&1 | grep -E "^\[|Error|Traceback"
    LV=$(ls -d $LD/REFINE_*/video.mp4 | head -1); echo "########## PERIODIC ours-2nd repaint s$SD ($(date)) ##########"; PYTHONPATH=$V1 python -m specwm.bench_base_prefix2 --ckpt $CK1 --image $IMG --action-path $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --frames 161 --steps 16 --prefix-latents 21 --mode repaint --seed $SD --prefix-video $LV --out $TD 2>&1 | grep -E "^\[|Error|Traceback"; fi
  echo "########## PERIODIC ours-2nd kvrw s$SD ($(date)) ##########"; cd $V2; PYTHONPATH=$V2 python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast --action-root $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --scene $SC --frames 161 --seeds $SD --base-video $BD/video.mp4 --configs W18_sink6 --arms kvrw_second_k10 --kv-providers "second=$TD/video.mp4" --out $O 2>&1 | grep -E "^\[|Error|Traceback"
done
touch $S/exp/Q143_${SHARD}_DONE; echo "[Q143_DONE shard $SHARD] $(date)"
