#!/bin/bash
S=$SCRATCH; module load gcc cuda python3 >/dev/null 2>&1
for V in $S/specwm-venv $S/SolarWM/.venv-solar; do
  source $V/bin/activate
  pip install -q av 2>&1 | grep -v notice | tail -1
  SP=$(python -c "import site; print(site.getsitepackages()[0])")
  rm -rf $SP/decord; cp -r $S/decord_shim/decord $SP/decord
  python - << 'PY'
import decord, io, numpy as np
p = "$SCRATCH/lingbot-world-v2/specwm_out/hands/base_p1/video.mp4"
vr = decord.VideoReader(p, ctx=decord.cpu(0))
print("  path:", len(vr), "frames", vr.get_avg_fps(), "fps", vr.get_batch([0,10,20]).asnumpy().shape)
decord.bridge.set_bridge("torch")
b = decord.VideoReader(io.BytesIO(open(p,"rb").read()), num_threads=1).get_batch([0,5])
print("  bytes+torch:", type(b).__name__, tuple(b.shape), b.dtype)
try:
    from vbench import VBench; print("  VBench import OK")
except Exception as e: print("  VBench:", type(e).__name__, str(e)[:100])
PY
  echo "--- $V 완료"; deactivate
done
