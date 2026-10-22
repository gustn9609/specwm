#!/bin/bash
# Item 3: expert swap (~38 s) 제거 — RePaint 16-step --resident (두 expert GPU 상주) vs swap, 480p 161 f / 73 f, S05 s1234 (E1' launder 출력 재사용)
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V1=$S/lingbot-world; V2=$S/lingbot-world-v2; CK=$V1/lingbot-world-base-cam; O=$R/resident; mkdir -p $O; cd $V1; export PYTHONPATH=$V1
SC=05; SD=1234; IMG=$V2/examples/$SC/image.jpg; LV161=$(ls -d $R/e1repro/S$SC/launder_s$SD/REFINE_*/video.mp4 | head -1); LV73=$(ls -d $R/window_time/launder_k6/REFINE_*/video.mp4 | head -1)
run() { echo "########## RESIDENT $1 ($(date)) ##########"; python -m specwm.bench_base_prefix2 --ckpt $CK --image $IMG --action-path $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --steps 16 --mode repaint --seed $SD --out $O/$1 "${@:2}" 2>&1 | grep -E "^\[|Error|Traceback|OutOfMemory"; }
run repaint161_resident --frames 161 --prefix-latents 9 --prefix-video $LV161 --resident
run repaint73_resident --frames 73 --prefix-latents 9 --prefix-video $LV73 --resident
run repaint73_swap --frames 73 --prefix-latents 9 --prefix-video $LV73
run repaint161_resident_360 --frames 161 --prefix-latents 9 --prefix-video $LV161 --resident --max-area 184320
touch $S/exp/Q125_DONE; echo "[Q125_DONE] $(date)"
