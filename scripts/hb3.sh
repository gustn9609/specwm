#!/bin/bash
S=$SCRATCH
date "+지금 %H:%M"; squeue -u $USER -h -o "%.9i %.10j %.9T %.8M"; true
sacct -j 968288,968367,968390 -X -o JobID,JobName%10,Elapsed,State -n
echo "=== base official ==="; grep -E "BASE official|BASE_OFFICIAL_DONE|gen_s|flicker|Error|Traceback" $S/base_official.968288.out 2>/dev/null | tail -10; true
echo "=== latency ==="; grep -E "^\[ours\] (GPTQ|\{)|LATENCY_DONE|실패" $S/lat_full.968367.out 2>/dev/null | tail -3; true
echo "=== solar s2 ==="; grep -vE "Warning|warn|^\s*$|Loading|nvpl|openmpi|Lmod|reloaded|^\s+[0-9]\)|config attributes" $S/solar_s2.968390.out 2>/dev/null | tail -12; true
echo "=== solar 출력 파일 ==="; find $S/solar_out/stage2_s05 -type f 2>/dev/null | head -20
