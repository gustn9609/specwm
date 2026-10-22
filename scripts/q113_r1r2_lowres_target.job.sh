#!/bin/bash
# R1+R2: 저해상도(360p 급, max_area 184320) target 교정 — launder(v1 low expert) + RePaint 16-step 를 320x560 에서 수행 → 픽셀 업샘플 → Fast KV-rewrite (resync_v4 provider 경로). E1'(480p) 와 recovery 비교 + 교정 시간.
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V1=$S/lingbot-world; V2=$S/lingbot-world-v2; CK=$V1/lingbot-world-base-cam; TAG=${TAG:-r1_360}; AREA=${AREA:-184320}; SCHED=${SCHED:-0.40,0.20,0.08}
O=$R/$TAG; mkdir -p $O
for SC in ${SCENES:-05 01 02}; do for SD in ${SEEDS:-1234 4321 9999}; do
  FV=$R/floor/fast/S$SC/W18_sink6/fast_alone_s$SD/video.mp4; BV=$R/floor/base/S${SC}_s$SD/video.mp4; IMG=$V2/examples/$SC/image.jpg
  [ -f $FV ] && [ -f $BV ] || { echo "[skip] S$SC s$SD inputs missing"; continue; }
  LD=$O/S$SC/launder_s$SD; TD=$O/S$SC/target_s$SD
  if [ ! -f $TD/video.mp4 ]; then
    echo "########## LAUNDER-LOWRES S$SC s$SD area=$AREA ($(date)) ##########"; cd $V1; T0=$(date +%s)
    PYTHONPATH=$V1 python -m specwm.hybrid_refine --ckpt $CK --scene $SC --image $IMG --action-dir $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --fast-video $FV --frames 153 --schedules "$SCHED" --max-area $AREA --out $LD
    T1=$(date +%s); LV=$(ls -d $LD/REFINE_*/video.mp4 | head -1)
    echo "########## REPAINT-LOWRES S$SC s$SD ($(date)) ##########"
    PYTHONPATH=$V1 python -m specwm.bench_base_prefix2 --ckpt $CK --image $IMG --action-path $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --frames 161 --steps 16 --prefix-latents 9 --mode repaint --seed $SD --prefix-video $LV --max-area $AREA --out $TD
    T2=$(date +%s); echo "{\"launder_s\": $((T1-T0)), \"repaint_s\": $((T2-T1))}" > $TD/correction_time.json; echo "[time] S$SC s$SD launder $((T1-T0)) s, repaint $((T2-T1)) s"
  fi
  echo "########## KVRW (upsampled provider) S$SC s$SD ($(date)) ##########"; cd $V2
  PYTHONPATH=$V2 python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast --action-root $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --scene $SC --frames 161 --seeds $SD --base-video $BV --configs W18_sink6 --arms kvrw_laundered --kv-providers "laundered=$TD/video.mp4" --out $O/S$SC
done; done
touch $S/exp/Q113_${TAG}_DONE; echo "[Q113_DONE $TAG] $(date)"
