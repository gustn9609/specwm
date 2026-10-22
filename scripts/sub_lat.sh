#!/bin/bash
S=$SCRATCH
true
cd $S/lingbot-world-v2 && python -c "import ast; ast.parse(open('specwm/latency.py').read()); print('서버 문법 OK')"
# 소규모 실행: 13프레임(1청크), fast_bf16 만 -> 호출 경로 검증
cat > $S/lat_smoke.slurm << 'SEOF'
#!/bin/bash
true
export SPECWM_NO_FLASH=1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD
python -m specwm.latency --ckpt $PWD/lingbot-world-v2-14b-causal-fast --action-root $SCRATCH/rt2_action_05 \
  --prompt-file $SCRATCH/p_05.txt --scene 05 --frames 25 --arms fast_bf16 --out $SCRATCH/lingbot-world-v2/specwm_out/latency_smoke
SEOF
sbatch $S/lat_smoke.slurm | tail -1
echo "--- vbench 설치 로그 ---"; tail -3 $S/vbench_install.log 2>/dev/null
squeue -u $USER -h -o "%.9i %.10j %.9T %.8M"
