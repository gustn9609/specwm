#!/bin/bash
S=$SCRATCH/lingbot-world
echo "=== generate.py 기본값 (frame_num, steps, guide, shift, solver) ==="
grep -nE "frame_num|sample_steps|sample_guide|sample_shift|sample_solver|default=" $S/generate.py | grep -iE "frame|step|guide|shift|solver" | head -20
echo; echo "=== README 길이/프레임 언급 ==="
grep -niE "frame|second|length|81|161" $S/README.md 2>/dev/null | head -15
echo; echo "=== v1 image2video.py 샘플링 루프 (430-480) ==="
sed -n 430,480p $S/wan/image2video.py
echo; echo "=== 내 bench_revisit Base 루프 ==="
grep -n "base" -i $S/specwm/bench_revisit.py | head -20
L=$(grep -n "def run_base\|model == .base\|if args.model" $S/specwm/bench_revisit.py | head -1 | cut -d: -f1)
[ -n "$L" ] && sed -n "${L},$((L+45))p" $S/specwm/bench_revisit.py
