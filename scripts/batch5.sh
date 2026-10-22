#!/bin/bash
S=$SCRATCH; module load gcc cuda python3 >/dev/null 2>&1
# 1) 기하 지표
cat > $S/geooff.slurm << 'SEOF'
#!/bin/bash
true
SEOF
sbatch $S/geooff.slurm | tail -1
# 2) VBench on official videos
sed -e 's#--out $SCRATCH/vbench_out #--out $SCRATCH/vbench_out2 #' -e '/--videos/,/^echo/{/=\$R/d}' $S/vbench.slurm > $S/vbench2.slurm
python3 - << 'PY'
p="$SCRATCH/vbench2.slurm"; s=open(p).read()
s=s.replace("--videos \\\n", "--videos \\\n  base_S05_official=$R/revisit/base_official_S05/video.mp4 \\\n  base_S01_official=$R/revisit/base_official_S01/video.mp4 \\\n  base_hands_official=$R/hands/base_official_p1/video.mp4\n")
s=s.replace("-o $SCRATCH/vbench.%j.out","-o $SCRATCH/vbench2.%j.out")
open(p,"w").write(s); print(open(p).read().split("python vbench_eval.py")[1][:400])
PY
sbatch $S/vbench2.slurm | tail -1
# 3) SolarWM 샘플 재생성 + 재제출
source $S/SolarWM/.venv-solar/bin/activate
rm -rf $S/solar_custom $S/solar_out/stage2_s05
cd $S && python solar_adapter.py --image $S/lingbot-world-v2/examples/05/image.jpg --poses $S/rt2_action_05/poses.npy --caption-file $S/p_05.txt --out $S/solar_custom --sample-id lingbot_s05 --pixel-frames 161 2>&1 | grep -E "selfcheck|translation_scale|Error"
sbatch $S/solar_s2.slurm | tail -1
squeue -u $USER -h -o "%.9i %.10j %.9T %.8M"
