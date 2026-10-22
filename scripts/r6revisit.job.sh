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

IMG=$V2/examples/00/image.jpg
PR=$V2/examples/00/prompt.txt          # 이미지에 맞는 정식 프롬프트 (기존 오염의 원인이었던 기본값 금지)
ACT=$R/revisit/yaw_action              # 0 -> 90 -> 0 도, 161 프레임, 평행이동 없음
OUT=$R/revisit2
mkdir -p $OUT

for f in "$IMG" "$PR"; do [ -e "$f" ] || { echo "MISSING $f"; exit 1; }; done
[ -f $ACT/poses.npy ] || { echo "MISSING $ACT/poses.npy"; exit 1; }
echo "[inputs] img=$IMG"
echo "[inputs] prompt=$(head -c 90 $PR)..."
echo "[inputs] action=$ACT"

# 161 프레임 -> 잠재 41 프레임 (인과 마스크에 필요)
export SPECWM_LAT_F=41

base() {  # $1=tag $2=steps $3=seed $4=causalN(0이면 양방향)
  local TAG=$1; local ST=$2; local SD=$3; local CN=$4
  if [ -f $OUT/$TAG/revisit.json ]; then echo "##### skip $TAG #####"; return 0; fi
  echo "##### $TAG : base steps=$ST seed=$SD causalN=$CN #####"
  cd $V1
  if [ "$CN" = "0" ]; then unset SPECWM_CAUSAL_FRAMES; else export SPECWM_CAUSAL_FRAMES=$CN; fi
  PYTHONPATH=$V1 python -m specwm.bench_base_official --ckpt $CKB \
    --image $IMG --action-path $ACT --prompt-file $PR \
    --frames 161 --steps $ST --seed $SD --out $OUT/$TAG || echo "!!! $TAG FAILED"
  unset SPECWM_CAUSAL_FRAMES
}

fast() {  # resync_v4 의 fast_alone (시드 지원)
  echo "##### fast_alone seeds=$1 #####"
  cd $V2
  PYTHONPATH=$V2 python -m specwm.resync_v4 --ckpt $CKF \
    --action-root $ACT --prompt-file $PR --scene 00 --frames 161 \
    --seeds $1 --configs W18_sink6 --arms fast_alone \
    --out $OUT/fast || echo "!!! fast FAILED"
}

# --- 1순위: 시드 1234 로 4개 arm (이게 핵심 질문에 답한다) ---
base base16_s1234   16 1234 0     # 기준 (고품질, 양방향)
base base4_s1234     4 1234 0     # 스텝 수만
base causalN3_s1234 16 1234 3     # 구조만 (Fast 와 같은 청크 크기)
fast 1234                          # 둘 다 + 증류

# --- 2순위: 두 번째 시드로 잡음 추정 ---
base base16_s4321   16 4321 0
base base4_s4321     4 4321 0
base causalN3_s4321 16 4321 3
fast 4321

echo "===== SUMMARY ====="
for d in $OUT/*/ ; do [ -f $d/revisit.json ] && echo "$(basename $d) : $(tr -d '\n ' < $d/revisit.json | cut -c1-200)"; done
ls -d $OUT/fast/*/ 2>/dev/null
echo "[R6REV_DONE]"
