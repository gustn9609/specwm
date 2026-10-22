#!/bin/bash
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; date "+지금 %H:%M"; squeue -u $USER -h -o "%.9i %.10j %.9T %.8M %.12R"; true
sacct -j 969727,969728,969729 -X -o JobID,JobName%9,Elapsed,State -n
echo "=== J-A ==="; grep -E "#####|\[prefix\]|chunk-causal|gen_s|JA_DONE|Error|Traceback|laundered:|SystemExit" $S/ja.969727.out 2>/dev/null | tail -20
ls $R/revisit | grep -E "e3"; ls $R/hybrid/s05_launder 2>/dev/null
echo "=== J-B ==="; grep -E "#####|gen_s|JB_DONE|Error|OutOfMemory" $S/jb.969728.out 2>/dev/null | tail -4
echo "=== J-C2 ==="; grep -cE "^\[W.*\] \{" $S/jc2.969729.out 2>/dev/null; grep -E "RESYNC_V2_DONE|실패" $S/jc2.969729.out 2>/dev/null | tail -2
