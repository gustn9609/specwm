#!/bin/bash
S=$SCRATCH
# ---- 실험 1: 다장면 창 스윕 ----
cat > $S/multi_scene.slurm << 'SEOF'
#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world-v2
export PYTHONPATH=$PWD
python -m specwm.multi_scene \
  --ckpt $PWD/lingbot-world-v2-14b-causal-fast \
  --action-root $SCRATCH --prompt-root $SCRATCH --frames 161 \
  --out $SCRATCH/lingbot-world-v2/specwm_out/multi_scene
SEOF
# ---- 실험 2: 하이브리드 (손 조작 = 소규모 예행 겸 물리 검증) ----
cat > $S/hybrid_hands.slurm << 'SEOF'
#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world
export PYTHONPATH=$PWD
python -m specwm.hybrid_refine \
  --ckpt $PWD/lingbot-world-base-cam --scene hands \
  --action-dir $SCRATCH/hands_action --prompt-file $SCRATCH/p1.txt \
  --fast-latent $SCRATCH/lingbot-world-v2/specwm_out/hands/fast_p1/latent.pt \
  --frames 81 --guide 5.0 \
  --schedules "0.25;0.15;0.25,0.10;0.40,0.20,0.08" \
  --out $SCRATCH/lingbot-world-v2/specwm_out/hybrid/hands
SEOF
sbatch $S/multi_scene.slurm | tail -1
sbatch $S/hybrid_hands.slurm | tail -1
squeue -u $USER -h -o "%.9i %.11j %.9T %.11S"
