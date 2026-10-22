#!/bin/bash
S=$SCRATCH; date "+지금 %H:%M"; sacct -j 968862 -X -o State,Elapsed -n
grep -E "SOLAR_S2_DONE" $S/solar_s2.968862.out 2>/dev/null
L=$(grep -n "solarwm: error" $S/solar_s2.968862.out 2>/dev/null | head -1 | cut -d: -f1)
if [ -n "$L" ]; then E=$((L+60)); sed -n "${L},${E}p" $S/solar_s2.968862.out | grep -E "File \".*solarwm|^\s+assert|AssertionError|^\s+[a-z_]+ = |^\s+raise" | grep -v elastic | tail -16; fi
find $S/solar_out/stage2_s05 -type f 2>/dev/null | grep -vE "json$" | head -8
