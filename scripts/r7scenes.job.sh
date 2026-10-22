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
OUT=$R/ablate2
mkdir -p $OUT
export SPECWM_LAT_F=41          # 161 프레임 -> 잠재 41

# --- A. few-step vs causal, 공식 예제 장면에서 (기존 hands 자작 프롬프트 대체) ---
# 이미 있는 것은 건너뜀: S05 base70/base16/base8/base4, S01 base70
bs() {   # $1=scene $2=tag $3=steps $4=causalN
  local SC=$1; local TAG=$2; local ST=$3; local CN=$4
  local D=$OUT/S${SC}_${TAG}
  if [ -f $D/revisit.json ]; then echo "##### skip S$SC $TAG #####"; return 0; fi
  echo "##### S$SC $TAG : steps=$ST causalN=$CN #####"
  cd $V1
  if [ "$CN" = "0" ]; then unset SPECWM_CAUSAL_FRAMES; else export SPECWM_CAUSAL_FRAMES=$CN; fi
  PYTHONPATH=$V1 python -m specwm.bench_base_official --ckpt $CKB \
    --image $V2/examples/$SC/image.jpg --action-path $S/rt2_action_$SC --prompt-file $S/p_$SC.txt \
    --frames 161 --steps $ST --seed 1234 --out $D || echo "!!! S$SC $TAG FAILED"
  unset SPECWM_CAUSAL_FRAMES
}
fa() {   # Fast 단독
  local SC=$1
  [ -f $OUT/S${SC}_fast/W18_sink6/fast_alone_s1234/frames_all.npy ] && { echo "##### skip S$SC fast #####"; return 0; }
  echo "##### S$SC fast_alone #####"
  cd $V2
  PYTHONPATH=$V2 python -m specwm.resync_v4 --ckpt $CKF --action-root $S/rt2_action_$SC \
    --prompt-file $S/p_$SC.txt --scene $SC --frames 161 --seeds 1234 --configs W18_sink6 \
    --arms fast_alone --out $OUT/S${SC}_fast || echo "!!! S$SC fast FAILED"
}

# 우선순위: 각 장면에서 빠진 요인 격리 arm 부터
bs 05 causalN3 16 3          # S05: 구조만 (base70/16/4 는 revisit/ 에 이미 있음)
bs 01 base16   16 0          # S01: 기준
bs 01 base4     4 0          # S01: 스텝만
bs 01 causalN3 16 3          # S01: 구조만
fa 05
fa 01
bs 05 causalN6 16 6          # 여유 있으면 청크 크기 의존성
bs 05 causalN12 16 12

# --- B. mask 대 RePaint: 어느 쪽이 더 좋은 KV 제공자인가 (다운스트림) ---
# 제공자는 prefix5 잡(981895)에서 나온 두 영상. 둘 다 같은 다듬은 과거·같은 시드에서 생성됨.
echo "##### mask vs repaint as KV provider (S05, W18_sink6, seed 1234) #####"
PROV="repaint=$R/prefix5/p9/video.mp4,mask=$R/prefix5/p9_mask/video.mp4"
if [ -f $R/prefix5/p9/video.mp4 ] && [ -f $R/prefix5/p9_mask/video.mp4 ]; then
  cd $V2
  PYTHONPATH=$V2 python -m specwm.resync_v4 --ckpt $CKF --action-root $S/rt2_action_05 \
    --prompt-file $S/p_05.txt --scene 05 --frames 161 --seeds 1234 --configs W18_sink6 \
    --arms fast_alone,kvrw_repaint,kvrw_mask --kv-providers "$PROV" \
    --out $OUT/provider || echo "!!! provider FAILED"
else
  echo "!!! prefix5 provider videos missing"
fi

echo "===== SUMMARY ====="
for d in $OUT/*/ ; do
  [ -f $d/revisit.json ] && echo "$(basename $d) : $(tr -d '\n ' < $d/revisit.json | cut -c1-190)"
done
find $OUT -name "frames_all.npy" | sed "s|$OUT/||" | sort
echo "[R7SC_DONE]"
