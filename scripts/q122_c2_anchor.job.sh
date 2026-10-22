#!/bin/bash
# Exp 3 (C2): 앵커 1프레임 교정 — Base(low expert σ0.40/0.20/0.08) 로 13 f 창(60..72, 조건 이미지 = Fast frame 60) launder 후 마지막 frame 72 만 앵커로 사용 (순수 1-frame I2V SDEdit 은 조건 이미지를 재생성해 무의미) → 그 프레임을 새 조건 이미지로 Fast 재시작(85 f). 3 scenes x 3 seeds.
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V1=$S/lingbot-world; V2=$S/lingbot-world-v2; CK=$V1/lingbot-world-base-cam; O=$R/c2_anchor; mkdir -p $O
for SC in 05 01 02; do for SD in 1234 4321 9999; do
  FV=$R/e1repro/S$SC/W18_sink6/fast_alone_s$SD/video.mp4; IMG=$V2/examples/$SC/image.jpg; LD=$O/S$SC/launder1f_s$SD
  if [ ! -d $LD ]; then echo "########## C2 ANCHOR SDEDIT (13 f window 60..72, image = Fast frame 60) S$SC s$SD ($(date)) ##########"; cd $V1; T0=$(date +%s)
    F60=$O/S$SC/fastframe60_s$SD.png; mkdir -p $O/S$SC/action60_s$SD; python -c "import imageio; from PIL import Image; import numpy as np; Image.fromarray(np.asarray([f for i,f in enumerate(imageio.get_reader('$FV')) if i==60][0])).save('$F60')"
    python -c "import numpy as np; np.save('$O/S$SC/action60_s$SD/poses.npy', np.load('$S/rt2_action_$SC/poses.npy')[60:73]); np.save('$O/S$SC/action60_s$SD/intrinsics.npy', np.load('$S/rt2_action_$SC/intrinsics.npy')[60:73])"
    PYTHONPATH=$V1 python -m specwm.hybrid_refine --ckpt $CK --scene $SC --image $F60 --action-dir $O/S$SC/action60_s$SD --prompt-file $S/p_$SC.txt --fast-video $FV --frame-offset 60 --frames 13 --schedules "0.40,0.20,0.08" --out $LD 2>&1 | grep -E "^\[|Error|Traceback"
    echo "[time] 1-frame sdedit S$SC s$SD $(( $(date +%s) - T0 )) s (incl. load)"; fi
done; done
for SC in 05 01 02; do for SD in 1234 4321 9999; do
  LD=$O/S$SC/launder1f_s$SD; NPY=$(ls $LD/REFINE_*/frames_all.npy | head -1); PNG=$O/S$SC/base1f_s$SD.png
  python -c "import numpy as np; from PIL import Image; Image.fromarray(np.load('$NPY')[-1]).save('$PNG')"
  echo "########## C2 FAST RESTART S$SC s$SD ($(date)) ##########"
  python $S/scripts/c2_anchor.py --scene $SC --seed $SD --anchors anchor_base1f=$PNG --out $O/S$SC 2>&1 | grep -E "^\[|Error|Traceback"
done; done
touch $S/exp/Q122_DONE; echo "[Q122_DONE] $(date)"
