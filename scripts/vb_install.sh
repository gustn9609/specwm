#!/bin/bash
S=$SCRATCH
true
LOG=$S/vbench_install2.log; : > $LOG
python -c "import numpy, timm, transformers; print('numpy', numpy.__version__, 'timm', timm.__version__, 'transformers', transformers.__version__)" >> $LOG 2>&1
for p in vbench pyiqa openai-clip easydict decord fvcore iopath omegaconf scikit-image tabulate termcolor yacs portalocker; do
  python -c "import importlib,sys; m='$p'.replace('-','_'); m={'openai_clip':'clip','scikit_image':'skimage'}.get(m,m); sys.exit(0 if importlib.util.find_spec(m) else 1)" 2>/dev/null && { echo "skip $p (있음)" >> $LOG; continue; }
  pip install --no-deps -q "$p" >> $LOG 2>&1 && echo "ok $p" >> $LOG || echo "FAIL $p" >> $LOG
done
python - >> $LOG 2>&1 << 'PY'
import importlib, traceback
for m in ["vbench", "pyiqa", "clip", "decord", "easydict", "fvcore", "omegaconf", "skimage"]:
    try: importlib.import_module(m); print("import OK", m)
    except Exception as e: print("import FAIL", m, "->", type(e).__name__, str(e)[:120])
try:
    from vbench import VBench; print("VBench 클래스 OK")
except Exception as e:
    print("VBench 클래스 FAIL:", type(e).__name__, str(e)[:200]); traceback.print_exc(limit=2)
PY
echo DONE >> $LOG
