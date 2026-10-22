#!/bin/bash
S=$SCRATCH
# J3 분석 (gh-dev 는 J2 가 점유 → gh 에 제출, 짧은 잡이라 backfill 로 잡히길 기대; 동시에 gh-dev 에도 큐잉)
cat > $S/j3.slurm << 'SEOF'
#!/bin/bash
true
SEOF
sbatch $S/j3.slurm | tail -1
# VBench 크롭 (로그인 노드)
true
OMP_NUM_THREADS=1 python $S/crop_vbench.py 2>&1 | grep -vE "WorkerThread" | tail -2
squeue -u $USER -h -o "%.9i %.10j %.9T %.8M"
