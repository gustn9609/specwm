#!/bin/bash
# Build the Matrix-Game 2.0 venv (official stack: torch 2.8 cu128 + flash-attn 2.8.1 wheel + torch.compile via triton 3.4).
# The ABot venv could not be reused: its torchvision (0.26.0+cu128) does not match its torch 2.8.0 ("operator torchvision::nms does not exist").
set -ex
UV=/home/hk28965/.local/bin/uv; V=/home/hk28965/specwm/tools/mg2/venv
$UV venv $V --python /usr/bin/python3.12 --seed
$UV pip install --python $V/bin/python torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
$UV pip install --python $V/bin/python /home/hk28965/specwm/tools/abot/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
# their requirements.txt minus the serving / TensorRT / ONNX / wandb / dashscope extras that inference.py never imports
$UV pip install --python $V/bin/python "diffusers==0.35.1" "transformers==4.57.1" "tokenizers>=0.20.3" "accelerate>=1.1.1" tqdm imageio imageio-ffmpeg easydict ftfy "numpy<2.3" omegaconf einops av safetensors opencv-python open_clip_torch sentencepiece pydantic scikit-image matplotlib "huggingface_hub[cli]" triton==3.4.0
$V/bin/python -c "import torch, torchvision, flash_attn, triton, diffusers, transformers; print('OK', torch.__version__, torchvision.__version__, flash_attn.__version__, triton.__version__, diffusers.__version__, transformers.__version__)"
echo BUILD_DONE
