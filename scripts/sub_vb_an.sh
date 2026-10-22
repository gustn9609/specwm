#!/bin/bash
S=$SCRATCH
sbatch $S/vb6.slurm | tail -1
# 분석은 base15 두 시드가 끝난 뒤 (jb3 970840) 시작하도록 의존성
cat > $S/an3.slurm << 'SEOF'
#!/bin/bash
true
export TORCH_HOME=$SCRATCH/.torch_hub; export HF_HUB_OFFLINE=1
cd $SCRATCH && python analysis_v3.py
SEOF
sbatch --dependency=afterany:970840 $S/an3.slurm | tail -1
squeue -u $USER -h -o "%.9i %.10j %.9T %.8M %.12R %.20E"
