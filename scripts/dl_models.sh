#!/bin/bash
cd /home/hk28965/specwm
python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
import os
for rid,ld in [('robbyant/lingbot-world-v2-14b-bid','lingbot-world-v2/lingbot-world-v2-14b-bid'),
               ('robbyant/lingbot-world-v2-14b-causal-pretrain','lingbot-world-v2/lingbot-world-v2-14b-causal-pretrain'),
               ('robbyant/lingbot-world-v2-1.3b-causal-fast','lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast/transformers')]:
    print('DL',rid,flush=True); snapshot_download(rid, local_dir=ld, max_workers=8); print('DONE',rid,flush=True)
hf_hub_download('DANNY621/H3-World','step-10000.safetensors',local_dir='h3world/checkpoints/H3-World'); print('DONE H3-World LoRA',flush=True)
print('DL Qwen3-VL',flush=True); snapshot_download('Qwen/Qwen3-VL-32B-Instruct', local_dir='hf/Qwen3-VL-32B-Instruct', max_workers=8); print('DONE Qwen3-VL',flush=True)
print('ALL_DL_DONE')
PY
