#!/bin/bash
date "+지금 %H:%M"
echo "=== 잡 상태 ==="
sacct -j 965661,965700,965701 -X -o JobID,JobName%11,Elapsed,State -n 2>/dev/null
echo
echo "=== 하이브리드 965701 ==="
grep -E '^\[FAST_asis|^\[REFINE|HYBRID_DONE|Traceback|SystemExit|Error' $SCRATCH/hybrid_hands.965701.out 2>/dev/null | tail -10
echo
echo "=== 다장면 965700 (완료 arm 수) ==="
grep -cE '^\[S[0-9]+_W' $SCRATCH/multi_scene.965700.out 2>/dev/null
grep -E '^\[S[0-9]+_W.*gen ' $SCRATCH/multi_scene.965700.out 2>/dev/null | tail -4
grep -E 'MULTI_SCENE_DONE|실패' $SCRATCH/multi_scene.965700.out 2>/dev/null | tail -3
echo
echo "=== 확장 창 스윕 965661 ==="
grep -E '^\[W[0-9]+.*\{|WIN_SWEEP_DONE|실패' $SCRATCH/win_sweep2.965661.out 2>/dev/null | tail -6
