#!/bin/bash
# sdvg_orig_launch.sh GPUS MODE [TAU] — runs the ORIGINAL SDVG pipeline (archive/dev) on our two prompts, 3 seeds, after the ported-SDVG sweep on GPUs 0-3 is completely done.
G=$1; M=$2; T=${3:--0.7}; Q=/home/hk28965/specwm/exp/queue; cd /home/hk28965/specwm/archive
until grep -q DL_DONE $Q/sdvg_download.log 2>/dev/null && [ -f checkpoints/krea-realtime-video-14b.safetensors ] && [ -f checkpoints/self_forcing_dmd.pt ]; do sleep 120; done
until grep -q "WORKER_DONE" $Q/w01_sdvg.log 2>/dev/null && grep -q "WORKER_DONE" $Q/w23_sdvg.log 2>/dev/null; do sleep 120; done
until [ -f /home/hk28965/specwm/paper/overnight/blind/batch4.json ]; do sleep 60; done     # post-sweep metrics + sheets (uses GPU 0) done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${G%%,*} | tr -d ' ')" -lt 2000 ] && [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${G##*,} | tr -d ' ')" -lt 2000 ]; do sleep 60; done
echo "########## SDVG_ORIG $M tau=$T on GPUs $G ($(date)) ##########"
MODEL_FOLDER=wan_models MODE=$M TAU=$T SEEDS=1234,4321,9999 NUM_BLOCKS=13 DIFFUSION_GPU=0 CUDA_VISIBLE_DEVICES=$G .venv/bin/python -u sdvg_ours.py 2>&1 | grep -E "SDVG_ORIG_DONE|Block [0-9]+:|Traceback|Error|OutOfMemory|line [0-9]+|Loaded|Video saved|Missing|Unexpected"
echo "[SDVG_ORIG_WORKER_DONE $M] $(date)"
