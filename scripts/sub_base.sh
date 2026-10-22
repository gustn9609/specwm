#!/bin/bash
S=$SCRATCH
true
cd $S/lingbot-world && python -c "import ast; ast.parse(open('specwm/bench_base_official.py').read()); print('서버 문법 OK')"
# Base 공식 샘플러: S05(프롬프트 일치) 먼저, 이어서 S01, 그리고 손 장면(정적, 대조군)
cat > $S/base_official.slurm << 'SEOF'
#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD
for SC in 05 01; do
  echo "########## BASE official, scene $SC ##########"
  python -m specwm.bench_base_official --ckpt $PWD/lingbot-world-base-cam \
    --image $SCRATCH/lingbot-world-v2/examples/$SC/image.jpg \
    --action-path $SCRATCH/rt2_action_$SC --prompt-file $SCRATCH/p_$SC.txt --frames 161 \
    --out $SCRATCH/lingbot-world-v2/specwm_out/revisit/base_official_S$SC
done
echo "########## BASE official, hands (static camera control) ##########"
python -m specwm.bench_base_official --ckpt $PWD/lingbot-world-base-cam \
  --image $SCRATCH/lingbot-world-v2/examples/hands/image.jpg \
  --action-path $SCRATCH/hands_action --prompt-file $SCRATCH/p_hands1.txt --frames 81 \
  --out $SCRATCH/lingbot-world-v2/specwm_out/hands/base_official_p1
SEOF
ls $S/hands_action/poses.npy $S/p_hands1.txt 2>&1 | head -3
sbatch $S/base_official.slurm | tail -1
# --- VBench 설치 시도 (로그인 노드, 백그라운드) ---
nohup bash -c "source $S/specwm-venv/bin/activate; pip install vbench > $S/vbench_install.log 2>&1; echo EXIT=\$? >> $S/vbench_install.log" > /dev/null 2>&1 &
echo "vbench 설치 백그라운드 시작"
squeue -u $USER -h -o "%.9i %.10j %.9T %.8M"
