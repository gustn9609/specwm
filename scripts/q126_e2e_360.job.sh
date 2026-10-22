#!/bin/bash
# Item 5: 360p draft + 360p target 지연 실측 — 교정 창 k=6 (73 f) 만 생성: launder(320x560, 3 fwd) + RePaint(320x560, 16-step, resident) → 360p 2-step draft 에 KV-rewrite (draft360_reground, arm kvrw_short) → recovery. 9 combos, 시간 기록.
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V1=$S/lingbot-world; V2=$S/lingbot-world-v2; CK=$V1/lingbot-world-base-cam; O=$R/e2e_360; mkdir -p $O; AREA=184320
for SC in 05 01 02; do for SD in 1234 4321 9999; do
  FV=$R/draft360_reground/S$SC/W18_sink6/fast_alone_s$SD/video.mp4; IMG=$V2/examples/$SC/image.jpg; LD=$O/S$SC/launder73_s$SD; TD=$O/S$SC/target73_s$SD
  [ -f $FV ] || { echo "[skip] no 360p draft S$SC s$SD"; continue; }
  if [ ! -f $TD/video.mp4 ]; then cd $V1; export PYTHONPATH=$V1
    echo "########## E2E-360 LAUNDER73 S$SC s$SD ($(date)) ##########"; T0=$(date +%s)
    python -m specwm.hybrid_refine --ckpt $CK --scene $SC --image $IMG --action-dir $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --fast-video $FV --frames 73 --schedules "0.40,0.20,0.08" --max-area $AREA --out $LD 2>&1 | grep -E "^\[|Error|Traceback"
    T1=$(date +%s); LV=$(ls -d $LD/REFINE_*/video.mp4 | head -1)
    echo "########## E2E-360 REPAINT73 (resident) S$SC s$SD ($(date)) ##########"
    python -m specwm.bench_base_prefix2 --ckpt $CK --image $IMG --action-path $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --frames 73 --steps 16 --prefix-latents 9 --mode repaint --seed $SD --prefix-video $LV --max-area $AREA --resident --out $TD 2>&1 | grep -E "^\[|Error|Traceback|OutOfMemory"
    T2=$(date +%s); echo "{\"launder_wall_s\": $((T1-T0)), \"repaint_wall_s\": $((T2-T1))}" > $TD/correction_time.json; echo "[time] S$SC s$SD launder73 $((T1-T0)) s repaint73 $((T2-T1)) s (wall incl. load)"
  fi
  echo "########## E2E-360 KVRW-SHORT S$SC s$SD ($(date)) ##########"; cd $V2; export PYTHONPATH=$V2
  python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast --action-root $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --scene $SC --frames 161 --seeds $SD --base-video $R/floor/base/S${SC}_s$SD/video.mp4 --configs W18_sink6 --max-area $AREA --nodes 0,3 \
    --arms kvrw_short --kv-providers "short=$TD/video.mp4" --out $R/draft360_reground/S$SC 2>&1 | grep -E "^\[|Error|Traceback"
done; done
touch $S/exp/Q126_DONE; echo "[Q126_DONE] $(date)"
