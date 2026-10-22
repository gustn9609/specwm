#!/bin/bash
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; date "+지금 %H:%M"
squeue -u $USER -h -o "%.9i %.8j %.9T %.9M %.14R"; echo
sacct -S now-6hours -X -o JobID,JobName%9,Elapsed,State -n | grep -vE "PENDING|RUNNING" | tail -6
echo "== E2 =="; f=$(ls -t $S/r4E2.*.out 2>/dev/null | head -1); grep -E "#####|BASE_PREFIX_DONE|R4E2_DONE|Error|Traceback|OutOfMemory" $f | cut -c1-100 | tail -8
echo "== E3 =="; f=$(ls -t $S/r4E3.*.out 2>/dev/null | head -1); [ -n "$f" ] && grep -E "#####|BASE_PREFIX_DONE|^\[W|실패|R4E3_DONE|Error|Traceback" $f | cut -c1-110 | tail -10
echo "== Aria =="; for f in $(ls -t $S/aria_H*.out 2>/dev/null); do echo "$f"; grep -E "#####|ARIA_BASE_DONE|Error" $f | tail -4; done
echo "== repro =="; for sc in 05 01 02; do echo "S$sc target=[$(for d in $R/repro/S$sc/target_s*; do [ -f $d/video.mp4 ] && echo x; done | wc -l)] base16=[$(for d in $R/repro/S$sc/base16_s*; do [ -f $d/video.mp4 ] && echo x; done | wc -l)] kvrw=[$(ls -d $R/repro/S$sc/W18_sink6/kvrw_* 2>/dev/null | wc -l)]"; done
ls $R/revisit/e3d_mask_refined 2>/dev/null | tr "\n" " "; echo
