#!/bin/bash
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out
true
python $S/mkslice2.py 05 720 73 2>/dev/null || python - << 'PY'
import numpy as np, os
S = os.environ["SCRATCH"]; src = S + "/traj55_action_05"; d = S + "/traj55_slice_f720_n73"; os.makedirs(d, exist_ok=True)
for f in ["poses.npy", "intrinsics.npy", "action.npy", "wasd_action.npy"]:
    a = np.load(os.path.join(src, f)); np.save(os.path.join(d, f), a[720:793].astype(np.float32))
print(d, np.load(d + "/poses.npy").shape)
PY
ls -d $S/traj55_slice*_n73 | tr '\n' ' '; echo
HDR='#!/bin/bash
true
export SPECWM_NO_FLASH=1; export OMP_NUM_THREADS=4; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V1=$S/lingbot-world; V2=$S/lingbot-world-v2; CK=$V1/lingbot-world-base-cam
ACT=$S/traj55_action_05; PR=$S/p_05.txt; CORR6=$R/e2e55/fixed/corr_k6/video.mp4
stage() { cd $V2; PYTHONPATH=$V2 python -m specwm.e2e_stage --ckpt $V2/lingbot-world-v2-14b-causal-fast --action-root $ACT --prompt-file $PR --scene 05 --frames 873 --seed 1234 --window 36 --sink 18 --state $1 --gen-until $2 --anchor fixed ${3:+--correction $3} $4; }
correct() {
  local ST=$1; local K=$2; local TAG=$3
  local F0=$(( (K-6)*12 ))
  local SL=$S/traj55_slice_f${F0}_n73
  echo "[correct] $TAG K=$K F0=$F0 SL=$SL"; [ -d "$SL" ] || { echo "slice missing $SL"; return 1; }
  local EX=e2e_S05_${TAG}_k$K; mkdir -p $V1/examples/$EX; cp $ST/prefix_k${K}_first.jpg $V1/examples/$EX/image.jpg
  cd $V1; PYTHONPATH=$V1 python -m specwm.hybrid_refine --ckpt $CK --scene $EX --action-dir $SL --prompt-file $PR --fast-latent $ST/prefix_k${K}_lat.pt --frames 33 --schedules "0.40,0.20,0.08" --out $ST/launder_k$K
  PYTHONPATH=$V1 python -m specwm.bench_base_prefix2 --ckpt $CK --image $ST/prefix_k${K}_first.jpg --action-path $SL --prompt-file $PR --frames 73 --steps 16 --prefix-latents 9 --mode repaint --seed 1234 --prefix-video $ST/launder_k$K/REFINE_0.40_0.20_0.08/video.mp4 --out $ST/corr_k$K
}
'
# ---- L: 현실적 지연 (Δ = 0, 1, 3, 7 청크 = 0 / 0.75 / 2.25 / 5.25 s), 단일 교정 고정 ----
cat > $S/r4L.slurm << SEOF
$HDR
E=\$R/lat55; mkdir -p \$E
for D in 0 1 3 7; do
  echo "##### latency delta = \$D chunks (\$(python -c "print(\$D*0.75)") s) #####"
  ST=\$E/d\$D; rm -rf \$ST; mkdir -p \$ST
  stage \$ST \$(( 6 + D )) "" ""
  stage \$ST 73 \$CORR6:0 --decode
done
echo "[R4L_DONE]"
SEOF
# ---- P: 교정 주기 (30 s = 청크 6,46 / 15 s = 6,26,46,66). '한 번만'은 L 의 D=0 재사용 ----
cat > $S/r4P.slurm << SEOF
$HDR
E=\$R/per55; mkdir -p \$E
echo "##### period 30 s: corrections at chunks 6, 46 #####"
ST=\$E/every30; rm -rf \$ST; mkdir -p \$ST
stage \$ST 6 "" ""
stage \$ST 46 \$CORR6:0 ""
correct \$ST 46 every30
stage \$ST 73 \$ST/corr_k46/video.mp4:40 --decode
echo "##### period 15 s: corrections at chunks 6, 26, 46, 66 #####"
ST=\$E/every15; rm -rf \$ST; mkdir -p \$ST
stage \$ST 6 "" ""
stage \$ST 26 \$CORR6:0 ""
correct \$ST 26 every15
stage \$ST 46 \$ST/corr_k26/video.mp4:20 ""
correct \$ST 46 every15
stage \$ST 66 \$ST/corr_k46/video.mp4:40 ""
correct \$ST 66 every15
stage \$ST 73 \$ST/corr_k66/video.mp4:60 --decode
echo "[R4P_DONE]"
SEOF
for J in L P; do bash -n $S/r4$J.slurm && echo "r4$J 문법 OK"; done
JL=$(sbatch --parsable $S/r4L.slurm | tail -1); JP=$(sbatch --parsable --dependency=afterany:$JL $S/r4P.slurm | tail -1)
echo "L=$JL P=$JP"; squeue -u $USER -h -o "%.9i %.8j %.9T %.9M %.14R"
