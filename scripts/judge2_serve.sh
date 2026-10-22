#!/bin/bash
# second Qwen3-VL-32B judge (vLLM) on GPU $1, port 30101 — same settings as the main judge
S=/home/hk28965/specwm; cd $S; export CUDA_VISIBLE_DEVICES=$1 HF_HOME=$S/hf VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME=$S/tools/judge-env/lib/python3.12/site-packages/nvidia/cu13; export PATH=$CUDA_HOME/bin:$PATH
exec tools/micromamba run -p tools/judge-env python -m vllm.entrypoints.openai.api_server --model hf/Qwen3-VL-32B-Instruct --served-model-name qwen3-vl-32b --port 30101 --host 127.0.0.1 --max-model-len 32768 --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":32,"video":2}'
