#!/bin/bash
set -x
V=/home/hk28965/specwm/tools/abot/venv; . $V/bin/activate
C=/home/hk28965/specwm/tools/cuda128; P=/home/hk28965/specwm/tools/pydev/usr/include/python3.12
NVINC=$(ls -d $V/lib/python3.12/site-packages/nvidia/*/include 2>/dev/null | tr '\n' ':')
export CUDA_HOME=$C PATH=$C/bin:$PATH CUDACXX=$C/bin/nvcc CMAKE_CUDA_COMPILER=$C/bin/nvcc CMAKE_CUDA_ARCHITECTURES=120 CUDAToolkit_ROOT=$C LIBRARY_PATH=$C/lib:$LIBRARY_PATH LD_LIBRARY_PATH=$C/lib:$LD_LIBRARY_PATH TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=32
export CPATH=$P:/home/hk28965/specwm/tools/pydev/usr/include/x86_64-linux-gnu/python3.12:/home/hk28965/specwm/tools/pydev/usr/include:$C/include:${NVINC}${CPATH}
cd /home/hk28965/specwm/tools/LightX2V/lightx2v_kernel && rm -rf build dist && CMAKE_BUILD_PARALLEL_LEVEL=32 uv build --wheel -Cbuild-dir=build . -Ccmake.define.CUTLASS_PATH=/home/hk28965/specwm/tools/cutlass -Ccmake.define.CMAKE_CUDA_COMPILER=$C/bin/nvcc -Ccmake.define.CUDAToolkit_ROOT=$C -Ccmake.define.Python_EXECUTABLE=$V/bin/python -Ccmake.define.Python_INCLUDE_DIR=$P -Ccmake.define.Python_INCLUDE_DIRS=$P --no-build-isolation > /home/hk28965/specwm/logs/abot_kernel_build3.log 2>&1; echo KERNEL_BUILD_EXIT $?; ls dist/; W=$(ls dist/*.whl 2>/dev/null | head -1); [ -n "$W" ] && uv pip install "$W" --force-reinstall --no-deps; (cd /tmp && python -c "import lightx2v_kernel; print('lightx2v_kernel ok')"); echo KERNEL_EXIT $?
uv pip install --no-build-isolation -v /home/hk28965/specwm/tools/SageAttention > /home/hk28965/specwm/logs/abot_sage_build3.log 2>&1; echo SAGE_BUILD_EXIT $?; (cd /tmp && python -c "import sageattention; print('sage ok')"); echo SAGE_EXIT $?
echo ABOT_BUILD3_DONE
