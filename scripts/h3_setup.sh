#!/bin/bash
# MiniMax-H3 용 venv: sglang[diffusion] (prerelease). 메인 venv 와 분리.
set -x; S=${SCRATCH:-/home/hk28965/specwm}; V=$S/h3-venv
[ -x $V/bin/pip ] || { python3 -m venv --without-pip $V; $V/bin/python $S/get-pip.py -q; }
source $V/bin/activate; pip install -q --upgrade pip uv
uv pip install --python $V/bin/python --prerelease=allow "sglang[diffusion]" 2>&1 | tail -5
python -c "import sglang, torch; print('sglang', sglang.__version__, 'torch', torch.__version__, torch.cuda.is_available())"
echo "[H3_SETUP_DONE]"
