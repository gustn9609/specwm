#!/bin/bash
S=$SCRATCH
true
LOG=$S/vbench_install3.log; : > $LOG
for V in $S/specwm-venv $S/SolarWM/.venv-solar; do
  source $V/bin/activate
  echo "### $V" >> $LOG
  python -c "import decord; print('decord 이미 있음', decord.__version__)" >> $LOG 2>&1 || pip install --no-deps -q eva-decord >> $LOG 2>&1 && echo "eva-decord 설치 시도" >> $LOG
  python -c "import decord; from decord import VideoReader; print('decord import OK', decord.__version__)" >> $LOG 2>&1 || echo "decord import FAIL" >> $LOG
  deactivate
done
source $S/specwm-venv/bin/activate
pip install --no-deps -q antlr4-python3-runtime==4.9.3 lazy_loader >> $LOG 2>&1
python - >> $LOG 2>&1 << 'PY'
import importlib
for m in ["vbench","decord","omegaconf","skimage","pyiqa","clip"]:
    try: importlib.import_module(m); print("import OK", m)
    except Exception as e: print("import FAIL", m, "->", type(e).__name__, str(e)[:100])
try:
    from vbench import VBench; print("VBench 클래스 OK")
except Exception as e: print("VBench 클래스 FAIL:", type(e).__name__, str(e)[:160])
PY
echo DONE >> $LOG
