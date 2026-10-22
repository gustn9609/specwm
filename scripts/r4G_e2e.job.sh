#!/bin/bash
true
export SPECWM_NO_FLASH=1; export OMP_NUM_THREADS=4; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V1=$S/lingbot-world; V2=$S/lingbot-world-v2; CK=$V1/lingbot-world-base-cam; E=$R/e2e55; mkdir -p $E
ACT=$S/traj55_action_05; PR=$S/p_05.txt; B16=$R/revisit/base_official_S05_s16/video.mp4
stage() { # $1 state, $2 gen_until, $3 anchor, $4 correction(optional "video:first"), $5 --decode(optional)
  cd $V2; PYTHONPATH=$V2 python -m specwm.e2e_stage --ckpt $V2/lingbot-world-v2-14b-causal-fast --action-root $ACT --prompt-file $PR --scene 05 --frames 873 --seed 1234 --window 36 --sink 18 --state $1 --gen-until $2 --anchor $3 ${4:+--correction $4} $5; }
correct() { # $1 state, $2 k (교정 적용 시점; prefix = 청크 k-6..k-4, 재생성 k-3..k-1) -> $1/corr_k$2/video.mp4 (73 f)
  local ST=$1 K=$2 F0=$(( (K-6)*12 )); local SL=$(python $S/mkslice.py $F0 73)
  mkdir -p $V1/examples/e2e_$(basename $ST)_k$K; cp $ST/prefix_k${K}_first.jpg $V1/examples/e2e_$(basename $ST)_k$K/image.jpg
  cd $V1; PYTHONPATH=$V1 python -m specwm.hybrid_refine --ckpt $CK --scene e2e_$(basename $ST)_k$K --action-dir $SL --prompt-file $PR --fast-latent $ST/prefix_k${K}_lat.pt --frames 33 --schedules "0.40,0.20,0.08" --out $ST/launder_k$K
  PYTHONPATH=$V1 python -m specwm.bench_base_prefix2 --ckpt $CK --image $ST/prefix_k${K}_first.jpg --action-path $SL --prompt-file $PR --frames 73 --steps 16 --prefix-latents 9 --mode repaint --seed 1234 --prefix-video $ST/launder_k$K/REFINE_0.40_0.20_0.08/video.mp4 --out $ST/corr_k$K; }
echo "##### G0: continuous fast_alone W36_sink18 (baseline) #####"
cd $V2; PYTHONPATH=$V2 python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast --action-root $ACT --prompt-file $PR --scene 05 --frames 873 --seeds 1234 --base-video $B16 --configs none --extra-configs "W36_sink18:fast_alone+resync_k6" --out $R/resync55
echo "##### G1: staged control (no corrections) #####"
ST=$E/control; stage $ST 6 fixed; stage $ST 26 fixed; stage $ST 46 fixed; stage $ST 73 fixed "" --decode
echo "##### G2: rolling anchors, corrections at k=6,26,46 #####"
ST=$E/rolling; stage $ST 6 rolling; correct $ST 6; stage $ST 26 rolling $ST/corr_k6/video.mp4:0; correct $ST 26; stage $ST 46 rolling $ST/corr_k26/video.mp4:20; correct $ST 46; stage $ST 73 rolling $ST/corr_k46/video.mp4:40 --decode
echo "##### G3: fixed anchors (first correction pinned), later corrections rewrite rolling window only #####"
ST=$E/fixed; mkdir -p $ST; cp $E/rolling/display_lat.pt $ST/ 2>/dev/null; rm -f $ST/*
stage $ST 6 fixed; cp -r $E/rolling/corr_k6 $ST/corr_k6; stage $ST 26 fixed $ST/corr_k6/video.mp4:0; correct $ST 26; stage $ST 46 fixed $ST/corr_k26/video.mp4:20; correct $ST 46; stage $ST 73 fixed $ST/corr_k46/video.mp4:40 --decode
echo "[R4G_DONE]"
