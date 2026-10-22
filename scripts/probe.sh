#!/bin/bash
cd $SCRATCH/SolarWM
echo "=== raw-WDS 관련 소스 파일 ==="
grep -rl 'tarfile\|webdataset\|raw-wds\|raw_wds' src/solarwm/ 2>/dev/null | head -6
echo
echo "=== tar 멤버 확장자 읽는 곳 (읽기만) ==="
grep -rnE '\.(mp4|jpg|png|npy|json|txt)' src/solarwm/data/ 2>/dev/null | grep -viE 'output|save|write' | head -14
echo
echo "=== 인덱스 행 키 ==="
grep -rhoE 'row\[.[a-z_]+.\]' src/solarwm/ 2>/dev/null | sort -u | head -20
echo
echo "=== Base x0_fn 본문 ==="
sed -n '103,124p' $SCRATCH/lingbot-world/specwm/binding_v1.py
