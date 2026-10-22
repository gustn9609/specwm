#!/bin/bash
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out
sacct -S now-6hours -X -o JobID,JobName%8,Elapsed,State -n | grep -vE "PENDING|RUNNING"
echo "== G 전체 마커 =="; f=$(ls -t $S/r4G.*.out | head -1); grep -nE "#####|E2E_STAGE_DONE|교정|앵커|BASE_PREFIX_DONE|^\[W.*\] \{|실패|Error|AssertionError|FileNotFound|stack" $f | cut -c1-130
echo "== e2e55 =="; for v in control rolling fixed; do echo "$v: $(ls $R/e2e55/$v 2>/dev/null | tr '\n' ' ')"; done; ls $R/resync55/W36_sink18 2>/dev/null | tr "\n" " "; echo
echo "== F2 =="; f=$(ls -t $S/r4F2.*.out 2>/dev/null | head -1); grep -E "#####|^\[W|실패|R4F2_DONE|Error|BASE_PREFIX_DONE|base\]" $f | cut -c1-110 | tail -16
echo "== anr4 =="; f=$(ls -t $S/anr4.*.out 2>/dev/null | head -1); grep -E "^  |ANALYSIS_R4_DONE|Error|미완" $f | grep -v warn | grep -vE "repro" | tail -40
