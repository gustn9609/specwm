#!/bin/bash
# LongLive (NVlabs) env + checkpoints. venv: ll-venv (torch 2.8.0 cu128, torchao 0.13.0). Wan2.2-TI2V-5B base + LongLive-2.0-5B generator + LongLive-1.3B.
set -x; cd /home/hk28965/specwm; mkdir -p longlive && cd longlive
[ -d LongLive ] || git clone -q --depth 1 https://github.com/NVlabs/LongLive LongLive
cd /home/hk28965/specwm
if [ ! -d ll-venv ]; then python3 -m venv --without-pip ll-venv && curl -sS https://bootstrap.pypa.io/get-pip.py | ll-venv/bin/python; fi
ll-venv/bin/pip install -q torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 && echo TORCH_OK
ll-venv/bin/pip install -q torchao==0.13.0 && echo TORCHAO_OK
grep -v "^#\|CLIP.git\|tensorrt\|wandb\|dashscope\|pycocotools\|lmdb\|dominate\|starlette" longlive/LongLive/requirements.txt > /tmp/ll_req.txt; ll-venv/bin/pip install -q -r /tmp/ll_req.txt && echo REQS_OK
ll-venv/bin/pip install -q flash-attn --no-build-isolation 2>&1 | tail -1; echo FLASH_EXIT=$?
ll-venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
import os; os.chdir('/home/hk28965/specwm/longlive')
for rid,ld,pat in [('Efficient-Large-Model/LongLive-2.0-5B','LongLive-2.0-5B',None),('Efficient-Large-Model/LongLive-1.3B','LongLive-1.3B',None),('Wan-AI/Wan2.2-TI2V-5B','Wan2.2-TI2V-5B',None),('Wan-AI/Wan2.1-T2V-1.3B','Wan2.1-T2V-1.3B',None)]:
    print('DL',rid,flush=True); snapshot_download(rid, local_dir=ld, max_workers=8); print('DONE',rid,flush=True)
print('LL_DL_DONE')
PY
ll-venv/bin/python -c "import torch,torchao;print(torch.__version__, torchao.__version__, torch.cuda.is_available())"; echo LL_SETUP_DONE
