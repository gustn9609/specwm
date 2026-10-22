#!/bin/bash
# restart the Qwen3-VL judge on the GPU the A/B run frees, the moment it finishes (other users' jobs grab free GPUs within seconds)
S=/home/hk28965/specwm; O=$S/lingbot-world-v2/specwm_out/df_ab
until grep -q GEN_DONE $O/chain.txt; do sleep 1; done
G=${JG:-7}; echo "judge on GPU $G"
cd $S; export CUDA_VISIBLE_DEVICES=$G HF_HOME=$S/hf
export CUDA_HOME=$S/tools/judge-env/lib/python3.12/site-packages/nvidia/cu13; export PATH=$CUDA_HOME/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
exec tools/micromamba run -p tools/judge-env python -m vllm.entrypoints.openai.api_server --model hf/Qwen3-VL-32B-Instruct --served-model-name qwen3-vl-32b --port 30100 --host 127.0.0.1 --max-model-len 32768 --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":32,"video":2}'
