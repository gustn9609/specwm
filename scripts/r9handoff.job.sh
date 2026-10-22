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
CKB=$V1/lingbot-world-base-cam
CKF=$V2/lingbot-world-v2-14b-causal-fast
PR=$S/p_05.txt
OUT=$R/handoff
mkdir -p $OUT

# ---------- E-A 1) Fast 가 청크 0..2 를 만들며 중간 상태 덤프 (rt2 궤적, resync_v4 fast_alone 과 같은 설정) ----------
if [ ! -f $OUT/fast_prefix/xt_node3.pt ]; then
  echo "##### E-A dump: Fast prefix (node3 sigma=0.625, node2 sigma=0.833) #####"
  cd $V2; PYTHONPATH=$V2 python -m specwm.fast_dump --ckpt $CKF --action-root $S/rt2_action_05 --prompt-file $PR \
    --scene 05 --frames 161 --seed 4321 --phase prefix --window 18 --sink 6 --out $OUT/fast_prefix || { echo "!!! dump FAILED"; exit 1; }
fi
# Base 가 쓸 33프레임 궤적 슬라이스 (rt2 앞 73프레임 == traj55 앞 73프레임, poses 동일 확인됨)
SL=$S/traj55_slice_f0_n73

# ---------- E-A 2) Base 저노이즈 expert 가 x_t 에서 이어받아 0 까지 ----------
ho() {  # $1=tag $2=xt file $3=schedule
  [ -f $OUT/$1/video.mp4 ] && { echo "##### skip $1 #####"; return 0; }
  echo "##### E-A handoff $1 : schedule $3 #####"
  cd $V1; PYTHONPATH=$V1 python -m specwm.base_handoff --ckpt $CKB --prompt-file $PR --mode handoff \
    --xt $2 --image $OUT/fast_prefix/first.jpg --action-dir $SL --frames 33 --schedule "$3" --out $OUT/$1 || echo "!!! $1 FAILED"
}
ho ho3       $OUT/fast_prefix/xt_node3.pt "0.625,0.20,0.08"     # 마지막 노드에서 핸드오프 + 정제 2회
ho ho3_1step $OUT/fast_prefix/xt_node3.pt "0.625"               # 핸드오프 한 번만 (정제 없음)
ho ho2       $OUT/fast_prefix/xt_node2.pt "0.833,0.40,0.20,0.08" # 한 노드 더 일찍 핸드오프

# ---------- E-A 3) 각 핸드오프 결과를 prefix 로 RePaint 청크 3..5 (현행 corr 경로와 동일) ----------
rp() {  # $1=tag
  [ -f $OUT/$1/corr/video.mp4 ] && { echo "##### skip repaint $1 #####"; return 0; }
  [ -f $OUT/$1/video.mp4 ] || { echo "!!! no handoff video for $1"; return 1; }
  echo "##### E-A repaint $1 #####"
  cd $V1; PYTHONPATH=$V1 python -m specwm.bench_base_prefix2 --ckpt $CKB --image $OUT/fast_prefix/first.jpg \
    --action-path $SL --prompt-file $PR --frames 73 --steps 16 --prefix-latents 9 --mode repaint --seed 4321 \
    --prefix-video $OUT/$1/video.mp4 --out $OUT/$1/corr || echo "!!! repaint $1 FAILED"
}
rp ho3; rp ho3_1step; rp ho2

# ---------- E-B 1) Fast 55초 스트림 덤프 (e2e control 과 같은 설정: W36 sink18) ----------
if [ ! -f $OUT/fast_stream/xt_node3.pt ]; then
  echo "##### E-B dump: Fast 55 s stream (73 chunks) #####"
  cd $V2; PYTHONPATH=$V2 python -m specwm.fast_dump --ckpt $CKF --action-root $S/traj55_action_05 --prompt-file $PR \
    --scene 05 --frames 873 --seed 1234 --phase stream --window 36 --sink 18 --out $OUT/fast_stream || echo "!!! stream dump FAILED"
fi

# ---------- E-B 2) Base 검사기: 3청크마다 한 번 forward -> x0 추정 -> Fast x0 와 거리 ----------
if [ -f $OUT/fast_stream/xt_node3.pt ] && [ ! -f $OUT/verify/verify.json ]; then
  echo "##### E-B verify #####"
  cd $V1; PYTHONPATH=$V1 python -m specwm.base_handoff --ckpt $CKB --prompt-file $PR --mode verify \
    --stream-dir $OUT/fast_stream --slice-root $S --sigma-start 0.625 --every 3 --out $OUT/verify || echo "!!! verify FAILED"
fi

echo "===== SUMMARY ====="
for t in ho3 ho3_1step ho2; do echo "$t : handoff=$([ -f $OUT/$t/video.mp4 ] && echo OK || echo MISSING)  corr=$([ -f $OUT/$t/corr/video.mp4 ] && echo OK || echo MISSING)"; done
echo "stream dump: $([ -f $OUT/fast_stream/xt_node3.pt ] && echo OK || echo MISSING)   verify: $([ -f $OUT/verify/verify.json ] && echo OK || echo MISSING)"
echo "[R9HO_DONE]"
