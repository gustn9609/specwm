#!/bin/bash
S=$SCRATCH
cd $S/lingbot-world-v2 && python3 -c "import ast; ast.parse(open('specwm/fill_ours.py').read())" && echo "서버 문법 OK"
# --- OURS 8칸 ---
cat > $S/fill_ours.slurm << 'SEOF'
#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world-v2
export PYTHONPATH=$PWD
python -m specwm.fill_ours \
  --ckpt $PWD/lingbot-world-v2-14b-causal-fast \
  --action-root $SCRATCH --prompt-root $SCRATCH --frames 161 --bits 6 \
  --out $SCRATCH/lingbot-world-v2/specwm_out/fill_ours
SEOF
# --- BASE 장면 01, 05 ---
cat > $S/fill_base.slurm << 'SEOF'
#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
cd $SCRATCH/lingbot-world
export PYTHONPATH=$PWD
for SC in 01 05; do
  echo "########## BASE scene $SC ##########"
  mkdir -p examples/$SC
  cp $SCRATCH/lingbot-world-v2/examples/$SC/image.jpg examples/$SC/ 2>/dev/null
  python -m specwm.bench_revisit --model base \
    --ckpt $PWD/lingbot-world-base-cam --scene $SC --steps 70 --guide 5.0 \
    --action-dir $SCRATCH/rt2_action_$SC --prompt-file $SCRATCH/p_$SC.txt \
    --out $SCRATCH/lingbot-world-v2/specwm_out/revisit/base_S$SC
done
SEOF
sbatch $S/fill_ours.slurm | tail -1
sbatch $S/fill_base.slurm | tail -1
# --- multi_scene 영상 생성 (서버 ffmpeg) ---
cat > $S/mkvid.py << 'PYEOF'
import numpy as np, glob, os, imageio
n = 0
for d in sorted(glob.glob("$SCRATCH/lingbot-world-v2/specwm_out/multi_scene/S*")):
    p, o = os.path.join(d, "frames_all.npy"), os.path.join(d, "video.mp4")
    if not os.path.exists(p) or os.path.exists(o): continue
    imageio.mimwrite(o, list(np.load(p)), fps=4, quality=8); n += 1
print("multi_scene 영상 %d개 생성" % n)
PYEOF
true
source $S/specwm-venv/bin/activate
python $S/mkvid.py
squeue -u $USER -h -o "%.9i %.10j %.9T"
