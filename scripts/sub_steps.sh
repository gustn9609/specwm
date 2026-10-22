#!/bin/bash
S=$SCRATCH
cat > $S/base_steps.slurm << 'SEOF'
#!/bin/bash
true
export SPECWM_NO_FLASH=1
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD
R=$SCRATCH/lingbot-world-v2/specwm_out
for ST in 4 8 16; do
  echo "########## BASE official S05, steps=$ST ##########"
  python -m specwm.bench_base_official --ckpt $PWD/lingbot-world-base-cam \
    --image $SCRATCH/lingbot-world-v2/examples/05/image.jpg --action-path $SCRATCH/rt2_action_05 \
    --prompt-file $SCRATCH/p_05.txt --frames 161 --steps $ST --out $R/revisit/base_official_S05_s$ST
done
for ST in 4 16; do
  echo "########## BASE official hands, steps=$ST ##########"
  python -m specwm.bench_base_official --ckpt $PWD/lingbot-world-base-cam \
    --image $SCRATCH/lingbot-world-v2/examples/hands/image.jpg --action-path $SCRATCH/hands_action \
    --prompt-file $SCRATCH/p_hands1.txt --frames 81 --steps $ST --out $R/hands/base_official_p1_s$ST
done
echo "[BASE_STEPS_DONE]"
SEOF
sbatch $S/base_steps.slurm | tail -1
date "+%H:%M"; squeue -u $USER -h -o "%.9i %.10j %.9T %.8M  start=%S"
echo "=== solar s1/s2 진행 ==="
for j in 969170 969171; do f=$(ls $S/solar_s[12].$j.out 2>/dev/null); [ -n "$f" ] && { grep -E "SOLAR_S[12]_DONE" $f; grep -E "solarwm: error" $f | cut -c1-200; }; done
find $S/solar_out -name video.mp4 2>/dev/null | grep -v partial
