#!/bin/bash
S=$SCRATCH; L=$S/lingbot-world; date "+지금 %H:%M"; squeue -u $USER -h -o "%.9i %.6j %.9T %.8M %.14R"
N=$(squeue -u $USER -h | wc -l); echo "queue count: $N"
if [ $N -le 2 ]; then JD=$(sbatch --parsable --dependency=afterany:971805 $S/r4D.slurm 2>&1 | tail -1); echo "D submitted: $JD"; fi
echo "== A =="; grep -E "^\[W|실패|R4A_DONE|Error|Traceback" $L/slurm-971803.out 2>/dev/null | cut -c1-150
echo "== B =="; grep -E "^\[W|#####|실패|R4B_DONE|Error|BASE_PREFIX_DONE|gen_s|Traceback" $L/slurm-971804.out 2>/dev/null | cut -c1-120 | tail -12
echo "== C =="; grep -E "#####|R4C_DONE|Error|Traceback" $L/slurm-971805.out 2>/dev/null | cut -c1-120 | tail -3
