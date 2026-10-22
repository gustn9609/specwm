#!/bin/bash
# B1 verifier env: b1-venv (torch cu128 + transformers>=4.53 for V-JEPA 2 + VGGT from GitHub). Models: facebook/vjepa2-vitl-fpc64-256, facebook/VGGT-1B
set -x; cd /home/hk28965/specwm
[ -d b1-venv ] || { python3 -m venv --without-pip b1-venv && curl -sS https://bootstrap.pypa.io/get-pip.py | b1-venv/bin/python; }
b1-venv/bin/pip install -q torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128 && echo TORCH_OK
b1-venv/bin/pip install -q "transformers>=4.53" accelerate safetensors huggingface_hub imageio imageio-ffmpeg opencv-python-headless matplotlib scipy einops && echo TF_OK
mkdir -p b1 && cd b1 && ( [ -d vggt ] || git clone -q --depth 1 https://github.com/facebookresearch/vggt ) && cd .. && b1-venv/bin/pip install -q -e b1/vggt 2>&1 | tail -1; echo VGGT_INSTALL_EXIT=$?
b1-venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
for r,d in [("facebook/vjepa2-vitl-fpc64-256","hf/vjepa2-vitl-fpc64-256"),("facebook/VGGT-1B","hf/VGGT-1B")]:
    print("DL",r,flush=True); snapshot_download(r, local_dir=d, max_workers=8); print("DONE",r,flush=True)
PY
b1-venv/bin/python -c "
import torch; from transformers import AutoModel, AutoVideoProcessor
m=AutoModel.from_pretrained('hf/vjepa2-vitl-fpc64-256'); print('VJEPA2_OK', type(m).__name__, hasattr(m,'predictor'))
from vggt.models.vggt import VGGT; print('VGGT_IMPORT_OK')"
echo B1_SETUP_DONE
