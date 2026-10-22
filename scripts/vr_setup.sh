#!/bin/bash
# VisionReward-Video 전용 venv (CogVLM2-video: transformers 4.42.4 고정, decord, pytorchvideo, triton)
# 메인 venv 와 분리 (transformers 버전 충돌). torch 는 메인과 같은 cu128 휠.
set -x
S=${SCRATCH:-/home/hk28965/specwm}
V=$S/vr-venv
[ -x $V/bin/pip ] || { python3 -m venv --without-pip $V; $V/bin/python $S/get-pip.py -q; }
source $V/bin/activate
pip install -q --upgrade pip
pip install -q torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -q "transformers==4.42.4" "numpy<2" decord==0.6.0 pytorchvideo==0.1.5 einops timm sentencepiece accelerate \
   imageio imageio-ffmpeg opencv-python-headless scipy triton huggingface_hub
python - << 'PY'
import torch, transformers, decord, pytorchvideo, triton
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "transformers", transformers.__version__, "triton", triton.__version__)
PY
echo "[VR_SETUP_DONE]"
