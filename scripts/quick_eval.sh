#!/bin/bash
# quick_eval.sh RUNNAME GPU  — motion/flicker/boundary, temporal (DINO) metrics, sharpness for the six test cases of a run name (<task>/<RUNNAME>_s<seed>)
R=$1; G=${2:-0}; V=lingbot-world-v2/specwm_out/spec_v0; cd /home/hk28965/specwm
D=""; for t in sand_filling_up_jar sandcastle_wave_impact; do for s in 1234 4321 9999; do D="$D $V/$t/${R}_s$s"; done; done
venv/bin/python scripts/motion_eval.py $D 2>&1 | grep -v "^ " | cut -c1-150
CUDA_VISIBLE_DEVICES=$G venv/bin/python scripts/temporal_eval.py $V/temporal_eval_$R.json $D 2>&1 | grep -v -i "warn\|cache" | cut -c1-200
venv/bin/python - "$R" <<'PY'
import imageio, numpy as np, cv2, sys
R=sys.argv[1]
def hf(f):
    g=cv2.cvtColor(f,cv2.COLOR_RGB2GRAY).astype(np.float32); return float(np.abs(cv2.Laplacian(g,cv2.CV_32F)).mean())
for task in ("sand_filling_up_jar","sandcastle_wave_impact"):
    for sd in (1234,4321,9999):
        out=[]
        for r in ("base70_s%d", R+"_s%d"):
            fr=[np.asarray(f) for f in imageio.get_reader("lingbot-world-v2/specwm_out/spec_v0/%s/%s/video.mp4"%(task,r%sd))]; n=len(fr); idx=[min(i,n-1) for i in range(0,161,20)]
            out.append(np.mean([hf(fr[i]) for i in idx]))
        print("sharp %-6s s%d  base %.2f  run %.2f"%(task[:6],sd,out[0],out[1]))
PY
