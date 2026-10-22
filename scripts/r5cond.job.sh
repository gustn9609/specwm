#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

S=$SCRATCH
R=$S/lingbot-world-v2/specwm_out
V1=$S/lingbot-world
V2=$S/lingbot-world-v2
CK=$V1/lingbot-world-base-cam

# --- 재사용 입력: S05, 첫 교정(k=6) 창 = 프레임 0..72 (청크 0..5) ---
SRC=$R/e2e55/rolling
IMG=$SRC/prefix_k6_first.jpg
LAU=$SRC/launder_k6/REFINE_0.40_0.20_0.08/video.mp4
SL=$S/traj55_slice_f0_n73
PR=$S/p_05.txt
OUT=$R/prefix5
mkdir -p $OUT

for f in "$IMG" "$LAU" "$PR"; do [ -e "$f" ] || { echo "MISSING $f"; exit 1; }; done
[ -d "$SL" ] || { echo "MISSING $SL"; exit 1; }
echo "[inputs] ok  img=$IMG  lau=$LAU  slice=$SL"

run() {
  local TAG=$1; shift
  if [ -f $OUT/$TAG/revisit.json ]; then echo "##### skip $TAG (done) #####"; return 0; fi
  echo "##### arm $TAG : $* #####"
  cd $V1
  PYTHONPATH=$V1 python -m specwm.bench_base_prefix2 \
    --ckpt $CK --image $IMG --action-path $SL --prompt-file $PR \
    --prefix-video $LAU --frames 73 --steps 16 --seed 1234 --full-frames \
    "$@" --out $OUT/$TAG || echo "!!! arm $TAG FAILED"
}

# 우선순위 순서 (타임아웃 대비)
run p9      --prefix-latents 9 --mode repaint                 # 현행 기준선 (전 프레임 저장으로 재실행)
run p3      --prefix-latents 3 --mode repaint                 # 1청크만 업로드
run p1      --prefix-latents 1 --mode repaint                 # 업로드 0 (장면 시작 이미지만)
run p9_u2   --prefix-latents 9 --mode repaint --resample 2    # RePaint resampling u=2
run p6      --prefix-latents 6 --mode repaint                 # 2청크 업로드
run p9_mask --prefix-latents 9 --mode mask                    # 마스크 조건화 (이음매 대조군)
run p9_u4   --prefix-latents 9 --mode repaint --resample 4    # resampling u=4

echo "===== SUMMARY ====="
for d in p9 p3 p1 p9_u2 p6 p9_mask p9_u4; do
  [ -f $OUT/$d/revisit.json ] && echo "$d : $(cat $OUT/$d/revisit.json | tr -d '\n ' | cut -c1-220)"
done
echo "[R5COND_DONE]"
