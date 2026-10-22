#!/bin/bash
S=$SCRATCH; date "+지금 %H:%M"; squeue -u $USER -h -o "%.9i %.10j %.9T %.8M  start=%S"; true
sacct -j 969087,969088 -X -o JobID,JobName%9,Elapsed,State -n
echo "=== attn 단위 테스트 ==="; grep -E "diff|ATTN_TEST|Error|assert" $S/attn_test.969087.out 2>/dev/null | tail -5
echo "=== solar s2 (969088) ==="; grep -E "SOLAR_S2_DONE" $S/solar_s2.969088.out 2>/dev/null
L=$(grep -n "solarwm: error" $S/solar_s2.969088.out 2>/dev/null | head -1 | cut -d: -f1)
if [ -n "$L" ]; then E=$((L+60)); sed -n "${L},${E}p" $S/solar_s2.969088.out | grep -E "solarwm: error|File \".*solarwm|^\s+assert|Error" | grep -v elastic | tail -10 | cut -c1-220; fi
find $S/solar_out/stage2_s05 -type f 2>/dev/null | grep -vE "json$" | head -10
