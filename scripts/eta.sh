#!/bin/bash
S=$SCRATCH
date "+지금 %H:%M"
echo "=== 큐 ==="
squeue -u $USER -h -o "%.9i %.10j %.9T %.8M %.10L" 2>/dev/null || echo "  (없음)"
echo
echo "=== 966274 fill_ours (8칸) ==="
grep -E "^\[OURS_S[0-9]+_W[0-9]+\] (시작|로드|\{|실패)" $S/fill_ours.966274.out 2>/dev/null | tail -6
echo "  완료 arm: $(grep -cE "^\[OURS_S[0-9]+_W[0-9]+\] \{" $S/fill_ours.966274.out 2>/dev/null)"
echo
echo "=== 966231 fill_base (2칸) ==="
grep -E "BASE scene|step [0-9]+/70|\[done\]|REVISIT_DONE" $S/fill_base.*.out 2>/dev/null | tail -4
echo
echo "=== 966271 geometric ==="
sacct -j 966271 -X -o State,Elapsed -n 2>/dev/null
grep -E "평균inlier|저장" $S/geo.966271.out 2>/dev/null | tail -3
