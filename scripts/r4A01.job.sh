#!/bin/bash
true
export SPECWM_NO_FLASH=1; export OMP_NUM_THREADS=4; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V1=$S/lingbot-world; V2=$S/lingbot-world-v2; CK=$V1/lingbot-world-base-cam
ACT=$S/traj55_action_01; PR=$S/p_01.txt; E=$R/e2e55_S01; mkdir -p $E
stage() { cd $V2; PYTHONPATH=$V2 python -m specwm.e2e_stage --ckpt $V2/lingbot-world-v2-14b-causal-fast --action-root $ACT --prompt-file $PR --scene 01 --frames 873 --seed 1234 --window 36 --sink 18 --state $1 --gen-until $2 --anchor $3 ${4:+--correction $4} $5; }
correct() {
  local ST=$1; local K=$2; local TAG=$3
  local F0=$(( (K-6)*12 ))
  local SL=$(python $S/mkslice2.py 01 $F0 73)
  echo "[correct] S01 $TAG K=$K F0=$F0 SL=$SL"; [ -d "$SL" ] || { echo "slice missing"; return 1; }
  local EX=e2e_S01_${TAG}_k$K; mkdir -p $V1/examples/$EX; cp $ST/prefix_k${K}_first.jpg $V1/examples/$EX/image.jpg
  cd $V1; PYTHONPATH=$V1 python -m specwm.hybrid_refine --ckpt $CK --scene $EX --action-dir $SL --prompt-file $PR --fast-latent $ST/prefix_k${K}_lat.pt --frames 33 --schedules "0.40,0.20,0.08" --out $ST/launder_k$K
  PYTHONPATH=$V1 python -m specwm.bench_base_prefix2 --ckpt $CK --image $ST/prefix_k${K}_first.jpg --action-path $SL --prompt-file $PR --frames 73 --steps 16 --prefix-latents 9 --mode repaint --seed 1234 --prefix-video $ST/launder_k$K/REFINE_0.40_0.20_0.08/video.mp4 --out $ST/corr_k$K
}
echo "##### S01 common: stream to 4.5 s, then the first correction #####"
B=$E/common; rm -rf $B $E/rolling $E/fixed; stage $B 6 fixed; correct $B 6 common
for V in rolling fixed; do
  echo "##### S01 variant: $V #####"
  ST=$E/$V; mkdir -p $ST; cp $B/display_lat.pt $B/kv_lat.pt $B/meta.json $ST/; cp $B/prefix_k6* $ST/; cp -r $B/corr_k6 $ST/corr_k6
  stage $ST 26 $V $ST/corr_k6/video.mp4:0
  correct $ST 26 $V
  stage $ST 46 $V $ST/corr_k26/video.mp4:20
  correct $ST 46 $V
  stage $ST 73 $V $ST/corr_k46/video.mp4:40 --decode
done
echo "[R4A01_DONE]"
