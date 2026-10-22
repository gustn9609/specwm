#!/bin/bash
cd /home/hk28965/specwm/tools/SageAttention
export CUDA_HOME=/home/hk28965/specwm/tools/cudahome PATH=/home/hk28965/specwm/tools/cudahome/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=16 NVCC_APPEND_FLAGS="--threads 4"
export CPATH=/home/hk28965/specwm/tools/pydev/usr/include/python3.12:/home/hk28965/specwm/tools/pydev/usr/include/x86_64-linux-gnu/python3.12:/home/hk28965/specwm/tools/pydev/usr/include
/home/hk28965/specwm/venv/bin/pip install --no-build-isolation -v . 2>&1 | tail -40
echo BUILD_EXIT $?
