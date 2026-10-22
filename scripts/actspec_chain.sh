#!/bin/bash
# 10/13: run the B (adaptive chunk length) benchmark + the remaining A cases on a GPU as soon as the PID currently using it exits.
# usage: scripts/actspec_chain.sh GPU WAIT_PID CMD...   (CMD is run with CUDA_VISIBLE_DEVICES=GPU through envrun; logs to stdout)
G=$1; P=$2; shift 2
while [ -n "$P" ] && kill -0 "$P" 2>/dev/null; do sleep 30; done
cd /home/hk28965/specwm && CUDA_VISIBLE_DEVICES=$G exec ./scripts/envrun.sh "$@"
