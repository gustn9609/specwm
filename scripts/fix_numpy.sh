#!/bin/bash
S=$SCRATCH; module load gcc cuda python3 >/dev/null 2>&1; source $S/specwm-venv/bin/activate
pip uninstall -y -q projectaria_tools 2>&1 | tail -1
pip install -q "numpy<2" 2>&1 | tail -1
python -c "import numpy, cv2, torch, imageio; print('numpy', numpy.__version__, '| cv2', cv2.__version__, '| torch', torch.__version__)"
python -c "import decord, vbench" 2>&1 | tail -1 && echo "vbench import OK"
deactivate
echo "== 별도 venv for aria =="; python3 -m venv $S/aria-venv && source $S/aria-venv/bin/activate && pip install -q --upgrade pip 2>&1 | tail -1; pip install -q projectaria_tools opencv-python-headless imageio imageio-ffmpeg 2>&1 | tail -1
python -c "from projectaria_tools.core import data_provider, calibration, mps; import numpy; print('aria-venv OK numpy', numpy.__version__)"
squeue -u $USER -h -o "%.9i %.6j %.9T %.8M"
