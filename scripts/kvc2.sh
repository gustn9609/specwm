#!/bin/bash
S=$SCRATCH
cp $S/lingbot-world-v2/specwm/kv_compat.py $S/lingbot-world/specwm/kv_compat.py
cat > $S/kvcompat2.slurm << 'SEOF'
#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
OUT=$SCRATCH/lingbot-world-v2/specwm_out/kv_compat
echo "########## stage fast (v2) ##########"
cd $SCRATCH/lingbot-world-v2 && PYTHONPATH=$PWD python -m specwm.kv_compat --stage fast \
  --ckpt $PWD/lingbot-world-v2-14b-causal-fast --action-root $SCRATCH/rt2_action_00 \
  --prompt-file $SCRATCH/p_00.txt --max-layers 8 --out $OUT || exit 1
echo "########## stage base (v1) ##########"
cd $SCRATCH/lingbot-world && mkdir -p examples/00 && cp $SCRATCH/lingbot-world-v2/examples/00/image.jpg examples/00/ 2>/dev/null
PYTHONPATH=$PWD python -m specwm.kv_compat --stage base \
  --ckpt $PWD/lingbot-world-base-cam --action-root $SCRATCH/rt2_action_00 \
  --prompt-file $SCRATCH/p_00.txt --max-layers 8 --out $OUT || exit 1
echo "########## stage compare ##########"
PYTHONPATH=$PWD python -m specwm.kv_compat --stage compare --out $OUT
SEOF
sbatch $S/kvcompat2.slurm | tail -1
squeue -u $USER -h -o "%.9i %.10j %.9T %.8M"
