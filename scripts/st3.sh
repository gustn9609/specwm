#!/bin/bash
S=$SCRATCH; date "+지금 %H:%M"; squeue -u $USER -h -o "%.9i %.10j %.9T %.8M"; true
sacct -j 969475,969528,969529 -X -o JobID,JobName%9,Elapsed,State -n
echo "=== J2 ==="; grep -E "#####|gen_s|J2_DONE|Error|Traceback|chunk-causal|\[prefix\]|BASE_PREFIX_DONE" $S/j2.969475.out 2>/dev/null | tail -14
echo "=== J3 ==="; grep -E "E1 pairs|ANALYSIS_V2_DONE|Error|Traceback" $S/j3.969528.out 2>/dev/null | tail -3
echo "=== vbench4 ==="; grep -c "^\[" $S/vbench4.969529.out 2>/dev/null; grep -E "VBENCH_EVAL_DONE|Error" $S/vbench4.969529.out 2>/dev/null | head -2
