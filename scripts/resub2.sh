#!/bin/bash
S=$SCRATCH
sed -i 's/--set data.num_workers=0/--set data.num_workers=1/' $S/solar_s2.slurm
grep -n "num_workers" $S/solar_s2.slurm
sbatch $S/solar_s2.slurm | tail -1
# 본실행: latency 3 arm, 161프레임, scene 05
cat > $S/lat_full.slurm << 'SEOF'
#!/bin/bash
true
export SPECWM_NO_FLASH=1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD
python -m specwm.latency --ckpt $PWD/lingbot-world-v2-14b-causal-fast --action-root $SCRATCH/rt2_action_05 \
  --prompt-file $SCRATCH/p_05.txt --scene 05 --frames 161 --arms fast_bf16,fast_kv4,ours \
  --out $SCRATCH/lingbot-world-v2/specwm_out/latency_full
SEOF
sbatch $S/lat_full.slurm | tail -1
squeue -u $USER -h -o "%.9i %.10j %.9T %.8M"
