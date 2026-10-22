#!/bin/bash
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; date "+지금 %H:%M"
squeue -u $USER -h -o "%.9i %.8j %.9T %.9M %.14R"; sacct -j 977632,977633 -X -o JobID,JobName%6,Elapsed,State -n
for sc in 01 02; do f=$(ls -t $S/r4A$sc.*.out 2>/dev/null | head -1); echo "== S$sc ($f)"; [ -n "$f" ] && grep -E "#####|\[correct\]|E2E_STAGE_DONE|R4A${sc}_DONE|Error|Traceback|slice missing|OutOfMemory" $f | cut -c1-115 | tail -14; echo "  출력: $(ls $R/e2e55_S$sc 2>/dev/null | tr '\n' ' ')"; for v in rolling fixed; do [ -f $R/e2e55_S$sc/$v/video.mp4 ] && echo "    $v/video.mp4 있음"; done; done
