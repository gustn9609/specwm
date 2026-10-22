#!/bin/bash
S=$SCRATCH
date "+지금 %H:%M"
echo "=== 큐 ==="
squeue -u $USER -h -o "%.9i %.10j %.9T %.8M" 2>/dev/null || echo "  (없음)"
echo "=== 상태 ==="
sacct -j 966753,966754,966274 -X -o JobID,JobName%10,Elapsed,State -n 2>/dev/null
echo
echo "=== win60 966753 ==="
grep -E "^\[W[0-9]+.*\] (창|로드|GPTQ|\{|실패)|WIN60_DONE" $S/win60.966753.out 2>/dev/null | tail -12
echo "  청크 진행:"; grep -E "청크 [0-9]+/74" $S/win60.966753.out 2>/dev/null | tail -2
echo
echo "=== kvcompat 966754 ==="
grep -vE "Warning|warn|Loading checkpoint|^\s*$|FutureWarning|autocast|float32|sdpa_patch|\[mem\]|\[binding\]" $S/kvcompat.966754.out 2>/dev/null | tail -25
echo
echo "=== fillours 966274 ==="
grep -cE "^\[OURS_S[0-9]+_W[0-9]+\] \{" $S/fill_ours.966274.out 2>/dev/null | sed "s/^/  완료 arm: /"
grep -E "^\[OURS_S[0-9]+_W[0-9]+\] 실패|FILL_OURS_DONE" $S/fill_ours.966274.out 2>/dev/null | tail -3
