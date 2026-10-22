#!/bin/bash
# MiniMax-H3 FL2VA 서버 (GPU 0-3, Ulysses4). 쿡북의 RTX PRO 6000 4장 레시피. 로그: logs/h3_serve.out
S=/home/hk28965/specwm; source $S/h3-venv/bin/activate; cd $S; export CUDA_HOME=$S/h3-venv/lib/python3.12/site-packages/nvidia/cu13; export PATH=$HOME/.local/bin:$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH; export CPATH=$HOME/.local/include/python3.12:$HOME/.local/include:$CPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HOME=$S/hf
sglang serve --model-path $S/h3/MiniMax-H3 --model-variant fl2va --num-gpus 4 --ulysses-degree 4 --encoder-parallel auto --performance-mode speed --port 30010
