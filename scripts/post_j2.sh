#!/bin/bash
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out
scancel 969528 969529; sleep 2
# E3b arm (gh-dev, 사용자당 1개 제한 → 먼저 이것)
sed -e 's/-m specwm.resync_v2 /-m specwm.resync_v2_e3b /' -e 's#--out $R/resync_v2#--prefix-video $R/revisit/base_prefix_S05_L9/video.mp4 --out $R/resync_v2#' -e 's/-J j1resync/-J e3b/' -e 's#j1.%j.out#e3b.%j.out#' -e 's/-t 02:00:00/-t 00:30:00/' $S/j1.slurm > $S/e3b.slurm
grep -nE "resync_v2_e3b|prefix-video" $S/e3b.slurm | cut -c1-140
sbatch $S/e3b.slurm | tail -1
# J3 + VBench 는 gh 에 그대로 두되 gh-dev 에도 큐잉 (먼저 잡히는 쪽 사용; 중복 실행 시 뒤의 것 취소)
sed -e 's/-p gh -t 00:40:00/-p gh-dev -t 00:40:00/' $S/j3.slurm > $S/j3dev.slurm; sbatch $S/j3dev.slurm | tail -1
sed -e 's/-p gh -t 01:30:00/-p gh-dev -t 01:30:00/' $S/vbench4.slurm > $S/vbench4dev.slurm; sbatch $S/vbench4dev.slurm | tail -1
squeue -u $USER -h -o "%.9i %.10j %.9T %.8M %.12R"
