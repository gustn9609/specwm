#!/bin/bash
S=$SCRATCH
echo "=== win60 W18 결과 ==="
grep -E "^\[W18\] \{" $S/win60.966753.out | head -1
echo "=== V1Binding 시그니처 ==="
grep -n "def __init__" $S/lingbot-world/specwm/binding_v1.py | head -3
sed -n "$(grep -n 'def __init__' $S/lingbot-world/specwm/binding_v1.py | head -1 | cut -d: -f1),+6p" $S/lingbot-world/specwm/binding_v1.py
echo "=== fill_ours 누락 칸 ==="
python3 -c "
import json; d=json.load(open('$S/lingbot-world-v2/specwm_out/fill_ours/fill_ours.json'))
print('  완료:', sorted(d['arms']))"
# 기하 지표 잡
cat > $S/geo60.slurm << 'SEOF'
#!/bin/bash
true
source $SCRATCH/venv/bin/activate
cd $SCRATCH && python geo_win60.py
SEOF
sbatch $S/geo60.slurm | tail -1
# OURS 마지막 칸 재제출 (스크립트가 기존 칸은 건너뜀)
sed -i 's/-t 03:00:00/-t 01:00:00/' $S/fill_ours.slurm
sbatch $S/fill_ours.slurm | tail -1
