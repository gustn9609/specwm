#!/bin/bash
S=$SCRATCH
# --- 실험 A: 55초 창 스윕 (5 arm) ---
cat > $S/win60.slurm << 'SEOF'
#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world-v2
export PYTHONPATH=$PWD
python -m specwm.win60 \
  --ckpt $PWD/lingbot-world-v2-14b-causal-fast \
  --action-root $SCRATCH/traj60_action \
  --prompt-file $SCRATCH/p_00.txt --frames 893 \
  --out $SCRATCH/lingbot-world-v2/specwm_out/win60
SEOF
# --- 실험 B: KV 교정 타당성 검사 ---
cat > $S/kvcompat.slurm << 'SEOF'
#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
cd $SCRATCH/lingbot-world-v2
export PYTHONPATH=$PWD
python -m specwm.kv_compat \
  --fast-ckpt $PWD/lingbot-world-v2-14b-causal-fast \
  --base-ckpt $SCRATCH/lingbot-world/lingbot-world-base-cam \
  --action-root $SCRATCH/rt2_action_00 \
  --prompt-file $SCRATCH/p_00.txt --frames 13 --max-layers 8 \
  --out $SCRATCH/lingbot-world-v2/specwm_out/kv_compat
SEOF
sbatch $S/win60.slurm | tail -1
sbatch $S/kvcompat.slurm | tail -1
squeue -u $USER -h -o "%.9i %.10j %.9T %.8M"
