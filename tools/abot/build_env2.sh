#!/bin/bash
set -x
V=/home/hk28965/specwm/tools/abot/venv; . $V/bin/activate
export CUDA_HOME=/home/hk28965/specwm/tools/cudahome PATH=/home/hk28965/specwm/tools/cudahome/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=32 NVCC_APPEND_FLAGS="--threads 4"
export CUDACXX=/home/hk28965/specwm/tools/cudahome/bin/nvcc CMAKE_CUDA_COMPILER=/home/hk28965/specwm/tools/cudahome/bin/nvcc CMAKE_CUDA_ARCHITECTURES=120
export CPATH=/home/hk28965/specwm/tools/pydev/usr/include/python3.12:/home/hk28965/specwm/tools/pydev/usr/include/x86_64-linux-gnu/python3.12:/home/hk28965/specwm/tools/pydev/usr/include
cd /home/hk28965/specwm/tools/LightX2V/lightx2v_kernel && rm -rf build && CMAKE_BUILD_PARALLEL_LEVEL=32 uv build --wheel -Cbuild-dir=build . -Ccmake.define.CUTLASS_PATH=/home/hk28965/specwm/tools/cutlass -Ccmake.define.CMAKE_CUDA_COMPILER=/home/hk28965/specwm/tools/cudahome/bin/nvcc --no-build-isolation > /home/hk28965/specwm/logs/abot_kernel_build.log 2>&1; echo KERNEL_BUILD_EXIT $?; ls dist/ 2>/dev/null; W=$(ls dist/*.whl 2>/dev/null | head -1); [ -n "$W" ] && uv pip install "$W" --force-reinstall --no-deps; (cd /tmp && python -c "import lightx2v_kernel; print('lightx2v_kernel ok')"); echo KERNEL_EXIT $?
uv pip install --no-build-isolation -v /home/hk28965/specwm/tools/SageAttention > /home/hk28965/specwm/logs/abot_sage_build.log 2>&1; echo SAGE_BUILD_EXIT $?; (cd /tmp && python -c "import sageattention; print('sage ok')"); echo SAGE_EXIT $?
echo ABOT_BUILD2_DONE
