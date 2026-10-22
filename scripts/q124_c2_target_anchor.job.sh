#!/bin/bash
# C2 follow-up: anchor = frame 72 of the E1' corrected target (launder+RePaint 16-step, 480p) and of the 360p target (r1_360, upsampled) -> Fast restart. Is a Base-consistent single frame enough?
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; O=$R/c2_anchor
for SC in 05 01 02; do for SD in 1234 4321 9999; do
  P480=$O/S$SC/target480_f72_s$SD.png; P360=$O/S$SC/target360_f72_s$SD.png
  python -c "
import imageio, numpy as np; from PIL import Image
for src,dst in [('$R/e1repro/S$SC/target_s$SD/video.mp4','$P480'),('$R/r1_360/S$SC/target_s$SD/video.mp4','$P360')]:
    f=[x for i,x in enumerate(imageio.get_reader(src)) if i==72][0]; im=Image.fromarray(np.asarray(f)); im=im.resize((832,464),Image.BICUBIC) if im.size!=(832,464) else im; im.save(dst)"
  echo "########## C2 TARGET ANCHOR S$SC s$SD ($(date)) ##########"
  python $S/scripts/c2_anchor.py --scene $SC --seed $SD --anchors anchor_target480=$P480,anchor_target360=$P360 --out $O/S$SC 2>&1 | grep -E "^\[|Error|Traceback"
done; done
touch $S/exp/Q124_DONE; echo "[Q124_DONE] $(date)"
