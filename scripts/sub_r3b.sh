#!/bin/bash
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out
true
cd $S/lingbot-world-v2 && python -c "import ast; ast.parse(open('specwm/resync_v3.py').read()); print('resync_v3 OK')"
# J-C1: cond 6 + xfade 3 (gh-dev)
PV="sep_self=$R/revisit/e3sep_self_mask/video.mp4,sep_other=$R/revisit/e3sep_otherseed_mask/video.mp4,repaint_fast=$R/revisit/e3a_repaint_fast/video.mp4,repaint_self=$R/revisit/e3a_repaint_self/video.mp4,causal12=$R/revisit/e3b_causal12_repaint_fast/video.mp4,laundered=$R/revisit/e3c_repaint_laundered/video.mp4"
sed -e 's/-m specwm.resync_v2 /-m specwm.resync_v3 /' -e "s#--seeds 1234,4321,9999#--seeds 1234 --configs W18_sink6 --xfade --prefix-videos $PV#" -e 's/-J j1resync/-J jc1/' -e 's#j1.%j.out#jc1.%j.out#' -e 's/-t 02:00:00/-t 00:50:00/' $S/j1.slurm > $S/jc1.slurm
grep -c "prefix-videos" $S/jc1.slurm; sbatch $S/jc1.slurm | tail -1
# J-C2 재제출 (gh)
sbatch $S/jc2.slurm | tail -1
# J-B seed 1234 재시도 + 9999 백업 (gh)
sed -e 's/for SD in 1234 4321; do/for SD in 1234 9999; do/' -e 's/-J jbBase20/-J jbBase20b/' -e 's#jb.%j.out#jb2.%j.out#' $S/jb.slurm > $S/jb2.slurm
sbatch $S/jb2.slurm | tail -1
squeue -u $USER -h -o "%.9i %.10j %.9T %.8M %.12R"
