#!/bin/bash
S=$SCRATCH/lingbot-world-v2
echo "===== v2 model: sink / max_attention_size / current_start 사용처 ====="
grep -rn "sink\|max_attention_size\|current_start" $S/wan/modules/*.py | grep -v "^.*#" | head -30
echo; echo "===== binding_v2: sink 인자 / _kv_size / _mk_cache ====="
grep -n "sink\|_kv_size\|def _mk_cache\|local_attn" $S/specwm/binding_v2.py | head -20
