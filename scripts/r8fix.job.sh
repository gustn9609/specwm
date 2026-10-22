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
ACT=$S/traj55_action_05
PR=$S/p_05.txt

# 지연/주기 스윕은 arm 당 시드 1개뿐이라 노이즈(0.090) 안에 전부 묻혀 있었다.
# 시드 4321, 9999 를 추가해 arm 간 차이가 실재하는지 판정한다.
# 각 시드는 자기 자신의 교정본을 만들어야 한다 (seed 1234 의 corr_k6 재사용 불가).

stage() {  # $1=state dir  $2=gen-until  $3=correction(video:firstchunk)  $4=extra  $5=seed
  cd $V2
  PYTHONPATH=$V2 python -m specwm.e2e_stage --ckpt $V2/lingbot-world-v2-14b-causal-fast \
    --action-root $ACT --prompt-file $PR --scene 05 --frames 873 --seed $5 \
    --window 36 --sink 18 --state $1 --gen-until $2 --anchor fixed ${3:+--correction $3} $4
}

correct() {  # $1=state dir  $2=chunk k  $3=tag
  local ST=$1
  local K=$2
  local TAG=$3
  local F0=$(( (K-6)*12 ))
  local SL=$S/traj55_slice_f${F0}_n73
  [ -f $ST/corr_k$K/video.mp4 ] && { echo "[correct] skip k=$K (done)"; return 0; }
  echo "[correct] $TAG K=$K F0=$F0 SL=$SL"
  [ -d "$SL" ] || { echo "slice missing $SL"; return 1; }
  local EX=r8_${TAG}_k$K
  mkdir -p $V1/examples/$EX
  cp $ST/prefix_k${K}_first.jpg $V1/examples/$EX/image.jpg
  cd $V1
  PYTHONPATH=$V1 python -m specwm.hybrid_refine --ckpt $CK --scene $EX --action-dir $SL \
    --prompt-file $PR --fast-latent $ST/prefix_k${K}_lat.pt --frames 33 \
    --schedules "0.40,0.20,0.08" --out $ST/launder_k$K
  PYTHONPATH=$V1 python -m specwm.bench_base_prefix2 --ckpt $CK --image $ST/prefix_k${K}_first.jpg \
    --action-path $SL --prompt-file $PR --frames 73 --steps 16 --prefix-latents 9 \
    --mode repaint --seed 1234 --prefix-video $ST/launder_k$K/REFINE_0.40_0.20_0.08/video.mp4 \
    --out $ST/corr_k$K
}

done_check() { [ -f $1/frames_all.npy ]; }

for SD in 4321 9999; do
  echo "############################## seed $SD ##############################"

  # --- 1) 대조군: 교정 없음 (각 시드의 기준선) ---
  ST=$R/lat55_s$SD/control
  if done_check $ST; then echo "##### skip control s$SD #####"; else
    echo "##### control (no correction) seed $SD #####"
    rm -rf $ST; mkdir -p $ST
    stage $ST 73 "" --decode $SD
  fi

  # --- 2) 지연 스윕: delta = 0, 3, 7 청크 ---
  for D in 0 3 7; do
    ST=$R/lat55_s$SD/d$D
    if done_check $ST; then echo "##### skip d$D s$SD #####"; continue; fi
    echo "##### latency delta = $D chunks, seed $SD #####"
    rm -rf $ST; mkdir -p $ST
    stage $ST 6 "" "" $SD                       # 청크 6 까지 (여기서 prefix_k6_* 가 저장됨)
    correct $ST 6 lat${D}s${SD}                 # 이 시드 자신의 교정본 생성
    if [ "$D" != "0" ]; then
      stage $ST $(( 6 + D )) "" "" $SD          # 교정이 도착하기 전까지 계속 생성 = 지연
    fi
    stage $ST 73 $ST/corr_k6/video.mp4:0 --decode $SD
  done

  # --- 3) 주기 스윕: 30 초마다 (청크 6, 46) ---
  ST=$R/per55_s$SD/every30
  if done_check $ST; then echo "##### skip every30 s$SD #####"; else
    echo "##### period 30 s, seed $SD #####"
    rm -rf $ST; mkdir -p $ST
    stage $ST 6 "" "" $SD
    correct $ST 6 p30s${SD}
    stage $ST 46 $ST/corr_k6/video.mp4:0 "" $SD
    correct $ST 46 p30s${SD}
    stage $ST 73 $ST/corr_k46/video.mp4:40 --decode $SD
  fi
done

echo "===== SUMMARY ====="
for SD in 4321 9999; do
  for d in $R/lat55_s$SD/*/ $R/per55_s$SD/*/; do
    [ -d "$d" ] && echo "$(echo $d | sed "s|$R/||") : $([ -f $d/frames_all.npy ] && echo OK || echo MISSING)"
  done
done
echo "[R8SD_DONE]"
