#!/bin/bash
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; date "+지금 %H:%M"
squeue -u $USER -h -o "%.9i %.8j %.9T %.9M %.14R"; N=$(squeue -u $USER -h -p gh-dev | wc -l); echo "gh-dev queue: $N"
if [ $N -le 2 ]; then JG=$(sbatch --parsable --dependency=afterany:972771 $S/r4G_e2e.slurm 2>&1 | tail -1); echo "G submitted: $JG"; fi
echo "== E2 =="; f=$(ls -t $S/r4E2.*.out | head -1); grep -E "#####|R4E2_DONE|Error|Traceback" $f | cut -c1-90 | tail -4
echo "== E3 =="; f=$(ls -t $S/r4E3.*.out 2>/dev/null | head -1); [ -n "$f" ] && grep -E "#####|BASE_PREFIX_DONE|^\[W|실패|R4E3_DONE|Error|Traceback" $f | cut -c1-110 | tail -8
echo "== Aria =="; for f in $(ls -t $S/aria_H*.out 2>/dev/null); do grep -E "ARIA_BASE_DONE|Error|OutOfMemory" $f | tail -6; done
ls -d $R/aria/*_base70_81f_s* 2>/dev/null | xargs -n1 basename | tr "\n" " "; echo
