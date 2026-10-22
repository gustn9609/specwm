#!/bin/bash
S=$SCRATCH
true
source $S/specwm-venv/bin/activate
cd $S && python mkact.py
cd $S/lingbot-world-v2 && python3 -c "
import ast
for f in ['specwm/win60.py','specwm/kv_compat.py']:
    ast.parse(open(f).read()); print('서버 문법 OK', f)"
# 소규모 실행: 13프레임 1 arm
cat > $S/smoke60.slurm << 'SEOF'
#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world-v2
export PYTHONPATH=$PWD
python - << 'PY'
import specwm.win60 as w
w.ARMS = [("W18", 18, None)]        # 소규모: 1 arm 만
import sys
sys.argv = ["x", "--ckpt", "%s/lingbot-world-v2-14b-causal-fast" % __import__("os").getcwd(),
            "--action-root", "$SCRATCH/traj60_action",
            "--prompt-file", "$SCRATCH/p_00.txt",
            "--frames", "13",
            "--out", "$SCRATCH/lingbot-world-v2/specwm_out/smoke60"]
w.main()
PY
SEOF
sbatch $S/smoke60.slurm | tail -1
