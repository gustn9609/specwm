#!/bin/bash
S=$SCRATCH
echo "=== v1(Base) 공식 카메라 전처리 ==="
grep -rniE "plucker|c2w|w2c|extrinsic|cam_pose|poses\.npy|intrinsic" $S/lingbot-world/generate.py $S/lingbot-world/wan/*.py $S/lingbot-world/wan/utils/*.py 2>/dev/null | head -30
echo
echo "=== v2(Fast) 공식 카메라 전처리 ==="
grep -rniE "plucker|c2w|w2c|extrinsic|cam_pose|poses\.npy|intrinsic" $S/lingbot-world-v2/generate.py $S/lingbot-world-v2/wan/*.py $S/lingbot-world-v2/wan/utils/*.py 2>/dev/null | head -30
echo
echo "=== 내 binding_v1._prepare_cam ==="
L=$(grep -n '_prepare_cam' $S/lingbot-world/specwm/binding_v1.py | head -1 | cut -d: -f1)
sed -n "${L},$((L+30))p" $S/lingbot-world/specwm/binding_v1.py
echo
echo "=== 내 binding_v2 카메라 부분 ==="
grep -n "cam\|pose\|plucker\|Plucker" $S/lingbot-world-v2/specwm/binding_v2.py | head -20
