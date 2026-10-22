#!/bin/bash
# run a LingBot specwm command with the standard environment (venv, NVML preload, SageAttention, compile headers), from lingbot-world-v2/
S=/home/hk28965/specwm; source $S/env.sh; cd $S/lingbot-world-v2
export OMP_NUM_THREADS=8 PYTHONPATH=$PWD:$S/tools/SageAttention
export CPATH=$HOME/.local/include/python3.12:$HOME/.local/include/x86_64-linux-gnu/python3.12:$HOME/.local/include:$S/tools/pydev/usr/include/python3.12:$S/tools/pydev/usr/include/x86_64-linux-gnu/python3.12:$S/tools/pydev/usr/include:$CPATH
exec "$@"
