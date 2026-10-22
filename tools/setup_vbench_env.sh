#!/bin/bash
# [10/16] separate venv for VBench (pins transformers 4.33 / numpy<2) + pyiqa; python 3.11 via uv
set -x; cd /home/hk28965/specwm/tools
/home/hk28965/.local/bin/uv venv vbench_env --python 3.11 || exit 1
/home/hk28965/.local/bin/uv pip install --python vbench_env/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128 || exit 1
/home/hk28965/.local/bin/uv pip install --python vbench_env/bin/python vbench imageio imageio-ffmpeg || exit 1
vbench_env/bin/python -c "import vbench, pyiqa, torch; print('VBENCH_ENV_OK', vbench.__file__, torch.__version__, torch.cuda.is_available())"
