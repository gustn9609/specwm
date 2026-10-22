#!/bin/bash
S=$SCRATCH; module load gcc cuda python3 >/dev/null 2>&1; source $S/specwm-venv/bin/activate
cd $S/lingbot-world-v2 && python -c "import ast; ast.parse(open('specwm/resync.py').read()); print('서버 문법 OK')"
grep -n "\"hw\"\|hw=" specwm/binding_v2.py | head -2
cat > $S/resync.slurm << 'SEOF'
#!/bin/bash
true
export SPECWM_NO_FLASH=1; export OMP_NUM_THREADS=4
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD
python -m specwm.resync --ckpt $PWD/lingbot-world-v2-14b-causal-fast --action-root $SCRATCH/rt2_action_05 \
  --prompt-file $SCRATCH/p_05.txt --scene 05 --frames 161 \
  --base-video $SCRATCH/lingbot-world-v2/specwm_out/revisit/base_official_S05/video.mp4 \
  --out $SCRATCH/lingbot-world-v2/specwm_out/resync_s05
SEOF
sbatch $S/resync.slurm | tail -1
date "+%H:%M"; squeue -u $USER -h -o "%.9i %.10j %.9T %.8M"
echo "=== base steps 진행 ==="; grep -E "BASE official|gen_s|BASE_STEPS_DONE" $S/base_steps.969187.out | tail -3
echo "=== solar s1 ==="; grep -E "SOLAR_S1_DONE|solarwm: error" $S/solar_s1.969196.out 2>/dev/null | cut -c1-160
