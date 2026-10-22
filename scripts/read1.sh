#!/bin/bash
S=$SCRATCH
echo "===== v1 sdpa_patch.py ====="; cat $S/lingbot-world/specwm/sdpa_patch.py
echo; echo "===== v1 attention 호출부 (model.py self_attn forward) ====="
grep -n "flash_attention(\|def forward\|window_size\|q_lens\|k_lens" $S/lingbot-world/wan/modules/model.py | head -20
echo; echo "===== binding_v1.prepare 전체 (34-72) ====="; sed -n 34,72p $S/lingbot-world/specwm/binding_v1.py
echo; echo "===== 공식 UniPC 루프 (image2video.py 520-580) ====="; sed -n 520,580p $S/lingbot-world/wan/image2video.py
