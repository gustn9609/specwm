#!/bin/bash
S=$SCRATCH; module load gcc cuda python3 >/dev/null 2>&1; source $S/SolarWM/.venv-solar/bin/activate
python $S/patch_attn.py
cat > $S/attn_test.slurm << 'SEOF'
#!/bin/bash
true
SEOF
sbatch $S/attn_test.slurm | tail -1
rm -rf $S/solar_out/stage2_s05; sbatch $S/solar_s2.slurm | tail -1
squeue -u $USER -h -o "%.9i %.10j %.9T %.8M"
