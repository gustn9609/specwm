#!/bin/bash
true
source $SCRATCH/venv/bin/activate
python - << 'PY'
import numpy as np, importlib
for m in ["cv2","scipy"]:
    try:
        x=importlib.import_module(m); print("  %-6s %s" % (m, getattr(x,"__version__","?")))
    except Exception as e: print("  %-6s 없음 (%s)" % (m, str(e)[:40]))
K=np.load("$SCRATCH/lingbot-world-v2/examples/00/intrinsics.npy")
print("\nintrinsics shape", K.shape, K.dtype)
print("첫 5행:\n", K[:5])
PY
