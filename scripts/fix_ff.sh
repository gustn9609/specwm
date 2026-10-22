#!/bin/bash
S=$SCRATCH
mkdir -p $S/bin && ln -sf $S/SolarWM/.venv-solar/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-aarch64-v7.0.2 $S/bin/ffmpeg
$S/bin/ffmpeg -version 2>&1 | head -1
# slurm 에 PATH 주입 (torchrun 앞)
grep -q 'export PATH=$SCRATCH/bin' $S/solar_s2.slurm || sed -i 's#^export PYTHONUNBUFFERED=1#export PYTHONUNBUFFERED=1\nexport PATH=$SCRATCH/bin:$PATH#' $S/solar_s2.slurm
grep -n "PATH\|torchrun" $S/solar_s2.slurm | head -3
rm -rf $S/solar_out/stage2_s05; sbatch $S/solar_s2.slurm | tail -1
