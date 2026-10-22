#!/bin/bash
# ABot-World extra native deps into tools/abot/venv (torch 2.8 cu128, py3.12): flash_attn 2.8.1 wheel, SageAttention (source, Blackwell sm120),
# lightx2v_kernel (LightX2V + cutlass). CUDA 12.8 toolchain from tools/cudahome; Python headers from tools/pydev (no system python3-dev).
set -x
V=/home/hk28965/specwm/tools/abot/venv; . $V/bin/activate
export CUDA_HOME=/home/hk28965/specwm/tools/cudahome PATH=/home/hk28965/specwm/tools/cudahome/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=32 NVCC_APPEND_FLAGS="--threads 4"
export CPATH=/home/hk28965/specwm/tools/pydev/usr/include/python3.12:/home/hk28965/specwm/tools/pydev/usr/include/x86_64-linux-gnu/python3.12:/home/hk28965/specwm/tools/pydev/usr/include
cd /home/hk28965/specwm/tools/abot
[ -f flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl ] || wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
uv pip install flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl && python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"; echo FLASH_EXIT $?
uv pip install ninja packaging setuptools wheel scikit-build-core pybind11
cd /home/hk28965/specwm/tools/SageAttention && pip install --no-build-isolation -v . 2>&1 | tail -5; python -c "import sageattention; print('sage ok')"; echo SAGE_EXIT $?
cd /home/hk28965/specwm/tools; [ -d cutlass ] || git clone -q --depth 1 https://github.com/NVIDIA/cutlass.git; [ -d LightX2V ] || git clone -q --depth 1 https://github.com/ModelTC/LightX2V.git
cd /home/hk28965/specwm/tools/LightX2V/lightx2v_kernel && CMAKE_BUILD_PARALLEL_LEVEL=32 uv build --wheel -Cbuild-dir=build . -Ccmake.define.CUTLASS_PATH=/home/hk28965/specwm/tools/cutlass --no-build-isolation 2>&1 | tail -15; uv pip install dist/*.whl --force-reinstall --no-deps; python -c "import lightx2v_kernel; print('lightx2v_kernel ok')"; echo KERNEL_EXIT $?
echo ABOT_BUILD_DONE
