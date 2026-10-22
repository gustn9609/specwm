#!/bin/bash
cd $SCRATCH/SolarWM
echo "=== archive.py 샘플 구성 (85-160행) ==="
sed -n '85,160p' src/solarwm/data/archive.py
echo
echo "=== IndexRow 정의 ==="
grep -rn 'class IndexRow' -A 25 src/solarwm/data/*.py | head -40
