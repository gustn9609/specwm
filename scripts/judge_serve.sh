#!/bin/bash
# Qwen3-VL-32B-Instruct OpenAI-compatible judge server (StEvo-Bench verifier backend; 논문은 Gemini 3.1 Pro) — usage: GPU=n bash scripts/judge_serve.sh
cd /home/hk28965/specwm; source h3-venv/bin/activate
S=/home/hk28965/specwm; export CUDA_HOME=$S/h3-venv/lib/python3.12/site-packages/nvidia/cu13; export PATH=$HOME/.local/bin:$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH; export CPATH=$HOME/.local/include/python3.12:$HOME/.local/include/x86_64-linux-gnu/python3.12:$HOME/.local/include:$CPATH; export HF_HOME=$S/hf
export CUDA_VISIBLE_DEVICES=${GPU:-7}
exec python -m sglang.launch_server --model-path hf/Qwen3-VL-32B-Instruct --served-model-name qwen3-vl-32b --port 30100 --host 127.0.0.1 --mem-fraction-static 0.85 --context-length 65536 --chat-template qwen2-vl --attention-backend triton --sampling-backend pytorch
