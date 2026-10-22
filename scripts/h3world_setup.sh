#!/bin/bash
set -x; cd /home/hk28965/specwm
mkdir -p h3world && cd h3world
[ -d H3-World ] || git clone -q --depth 1 https://github.com/Danzer1xxxxChan/H3-World H3-World
if [ ! -d DiffSynth-Studio-h3-v2 ]; then
  git clone -q https://github.com/modelscope/DiffSynth-Studio.git DiffSynth-Studio-h3-v2
  git -C DiffSynth-Studio-h3-v2 checkout -q "$(cat H3-World/code/diffsynth_base_commit.txt)"
  git -C DiffSynth-Studio-h3-v2 apply ../H3-World/code/diffsynth_h3_action.patch && echo PATCH_OK
fi
mkdir -p DiffSynth-Studio-h3-v2/models/MiniMax; ln -sfn /home/hk28965/specwm/h3/MiniMax-H3 DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3
cd /home/hk28965/specwm
if [ ! -d h3w-venv ]; then
  python3 -m venv --without-pip h3w-venv && curl -sS https://bootstrap.pypa.io/get-pip.py | h3w-venv/bin/python
fi
h3w-venv/bin/pip install -q torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128 && echo TORCH_OK
h3w-venv/bin/pip install -q -r h3world/H3-World/requirements.txt && echo REQS_OK
h3w-venv/bin/pip install -q -e h3world/DiffSynth-Studio-h3-v2 2>&1 | tail -2; echo DIFFSYNTH_INSTALL_EXIT=$?
h3w-venv/bin/python -c "import torch,diffsynth;print(torch.__version__,torch.cuda.is_available());from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline;print('PIPE_OK')"
echo H3W_SETUP_DONE
