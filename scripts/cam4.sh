#!/bin/bash
S=$SCRATCH/lingbot-world
echo "=== 내 bench_revisit Base 샘플링 루프 (140-200) ==="; sed -n 140,200p $S/specwm/bench_revisit.py
echo; echo "=== 공식 cfg i2v-A14B 기본값 ==="
grep -rnE "sample_steps|sample_guide_scale|sample_shift|boundary|frame_num" $S/wan/configs/*.py | head -12
echo; echo "=== 공식 UniPC step 호출 (480-520) ==="; sed -n 480,520p $S/wan/image2video.py
echo; echo "=== generate.py 단일 GPU/오프로드 옵션 ==="
grep -nE "offload_model|t5_cpu|convert_model_dtype|ulysses|dit_fsdp" $S/generate.py | head -10
