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
