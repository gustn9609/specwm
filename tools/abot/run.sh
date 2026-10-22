#!/bin/bash
# run an ABot-World command in its own venv (torch 2.8 cu128, flash-attn 2.8.1, SageAttention, lightx2v_kernel) with the headers Triton's JIT needs.
# usage: CUDA_VISIBLE_DEVICES=g tools/abot/run.sh python scripts/inference.py ...   (cd's into tools/abot)
S=/home/hk28965/specwm; cd $S/tools/abot; . venv/bin/activate
export CPATH=$S/tools/pydev/usr/include/python3.12:$S/tools/pydev/usr/include/x86_64-linux-gnu/python3.12:$S/tools/pydev/usr/include:$S/tools/cuda128_tk/include:$CPATH
export LD_LIBRARY_PATH=$S/tools/cuda128_tk/lib64:$LD_LIBRARY_PATH LIBRARY_PATH=$S/tools/cuda128_tk/lib64:$S/tools/cuda128_tk/lib64/stubs:$LIBRARY_PATH CUDA_HOME=$S/tools/cuda128_tk PATH=$S/tools/cuda128_tk/bin:$PATH OMP_NUM_THREADS=8
exec "$@"
