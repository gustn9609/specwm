#!/bin/bash
# Trajectory-distribution experiment: S05, many seeds x {Base70 (target), Fast (draft), Fast+ours (speculative), ideal}. SEEDS="..." per GPU shard.
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V1=$S/lingbot-world; V2=$S/lingbot-world-v2; CK1=$V1/lingbot-world-base-cam; SC=05; IMG=$V2/examples/$SC/image.jpg; A="--image $IMG --action-path $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --frames 161"; O=$R/spec_dist/S$SC; mkdir -p $O
for SD in $SEEDS; do
  BD=$R/floor/base/S${SC}_s$SD
  if [ ! -f $BD/video.mp4 ]; then echo "########## SPECDIST BASE s$SD ($(date)) ##########"; cd $V1; PYTHONPATH=$V1 python -m specwm.bench_base_official --ckpt $CK1 $A --seed $SD --out $BD 2>&1 | grep -E "^\[|Error|Traceback"; fi
  if [ ! -f $O/W18_sink6/fast_alone_s$SD/latent.pt ]; then echo "########## SPECDIST FAST s$SD ($(date)) ##########"; cd $V2; PYTHONPATH=$V2 python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast --action-root $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --scene $SC --frames 161 --seeds $SD --base-video $BD/video.mp4 --configs W18_sink6 --arms fast_alone,resync_k6 --out $O 2>&1 | grep -E "^\[|Error|Traceback"; fi
  LD=$O/launder_s$SD; TD=$O/target_s$SD
  if [ ! -f $TD/video.mp4 ]; then
    echo "########## SPECDIST LAUNDER s$SD ($(date)) ##########"; cd $V1; PYTHONPATH=$V1 python -m specwm.hybrid_refine --ckpt $CK1 --scene $SC --image $IMG --action-dir $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --fast-latent $O/W18_sink6/fast_alone_s$SD/latent.pt --frames 153 --schedules "0.40,0.20,0.08" --out $LD 2>&1 | grep -E "^\[|Error|Traceback"
    LV=$(ls -d $LD/REFINE_*/video.mp4 | head -1); echo "########## SPECDIST REPAINT s$SD ($(date)) ##########"; PYTHONPATH=$V1 python -m specwm.bench_base_prefix2 --ckpt $CK1 $A --steps 16 --prefix-latents 9 --mode repaint --seed $SD --prefix-video $LV --out $TD 2>&1 | grep -E "^\[|Error|Traceback"
  fi
  echo "########## SPECDIST KVRW s$SD ($(date)) ##########"; cd $V2; PYTHONPATH=$V2 python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast --action-root $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --scene $SC --frames 161 --seeds $SD --base-video $BD/video.mp4 --configs W18_sink6 --arms kvrw_laundered --kv-providers "laundered=$TD/video.mp4" --out $O 2>&1 | grep -E "^\[|Error|Traceback"
done
touch $S/exp/Q142_${SHARD}_DONE; echo "[Q142_DONE shard $SHARD] $(date)"
