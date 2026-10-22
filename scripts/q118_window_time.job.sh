#!/bin/bash
# Exp 2 (time side): 교정 창 k 청크에 필요한 target 길이(12k+1 f)로 launder + RePaint 시간 측정 (S05 s1234, 480p). k=6/3/2/1 -> 73/37/25/13 f
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V1=$S/lingbot-world; V2=$S/lingbot-world-v2; CK=$V1/lingbot-world-base-cam; O=$R/window_time; mkdir -p $O; cd $V1; export PYTHONPATH=$V1
SC=05; SD=1234; FV=$R/floor/fast/S$SC/W18_sink6/fast_alone_s$SD/video.mp4; IMG=$V2/examples/$SC/image.jpg
for KF in "6 73 9" "3 37 5" "2 25 3" "1 13 2"; do set -- $KF; K=$1; F=$2; P=$3
  echo "########## WINDOW TIME k=$K frames=$F ($(date)) ##########"; T0=$(date +%s)
  python -m specwm.hybrid_refine --ckpt $CK --scene $SC --image $IMG --action-dir $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --fast-video $FV --frames $F --schedules "0.40,0.20,0.08" --out $O/launder_k$K 2>&1 | grep -E "^\[|Error|Traceback"
  T1=$(date +%s); LV=$(ls -d $O/launder_k$K/REFINE_*/video.mp4 | head -1)
  python -m specwm.bench_base_prefix2 --ckpt $CK --image $IMG --action-path $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --frames $F --steps 16 --prefix-latents $P --mode repaint --seed $SD --prefix-video $LV --out $O/target_k$K 2>&1 | grep -E "^\[|Error|Traceback"
  T2=$(date +%s); echo "{\"k\": $K, \"frames\": $F, \"launder_s\": $((T1-T0)), \"repaint_s\": $((T2-T1))}" > $O/time_k$K.json; echo "[time] k=$K frames=$F launder $((T1-T0)) s repaint $((T2-T1)) s"
done
touch $S/exp/Q118_DONE; echo "[Q118_DONE] $(date)"
