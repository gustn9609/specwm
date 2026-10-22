#!/bin/bash
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; date "+지금 %H:%M"
squeue -u $USER -h -o "%.9i %.8j %.9T %.9M %.14R"; N=$(squeue -u $USER -h -p gh-dev | wc -l); echo "gh-dev queue: $N"
sacct -S now-9hours -X -o JobID,JobName%9,Elapsed,State -n | grep -vE "PENDING|RUNNING" | tail -8
echo "== E3 =="; f=$(ls -t $S/r4E3.*.out 2>/dev/null | head -1); [ -n "$f" ] && grep -E "#####|R4E3_DONE|실패|Error|Traceback" $f | cut -c1-110 | tail -8; grep -cE "^\[W.*\] \{" $f 2>/dev/null
echo "== F1 =="; f=$(ls -t $S/r4F1.*.out 2>/dev/null | head -1); [ -n "$f" ] && grep -E "#####|R4F1_DONE|실패|Error|Traceback|BASE_PREFIX_DONE" $f | cut -c1-110 | tail -8; grep -cE "^\[W.*\] \{" $f 2>/dev/null
echo "== 결과 디렉토리 =="; ls $R/revisit/e3d_mask_refined 2>/dev/null | tr "\n" " "; echo; for sc in 05 01 02; do echo "S$sc target=[$(for d in $R/repro/S$sc/target_s*; do [ -f $d/video.mp4 ] && echo x; done | wc -l)] kvrw_ok=[$(for d in $R/repro/S$sc/W18_sink6/kvrw_*; do [ -f $d/video.mp4 ] && echo x; done | wc -l)]"; done
ls $R/lockon/W18_sink6/ | tr "\n" " "; echo; ls $R/resync55/ $R/resync55/W24_sink18 2>/dev/null | tr "\n" " "; echo; ls $R/targetcost_kv/W18_sink6 $R/targetcost_res 2>/dev/null | tr "\n" " "; echo
echo "== Aria =="; ls -d $R/aria/*_base70_81f_s* | wc -l; grep -hE "ARIA_BASE_DONE" $S/aria_H*.out | wc -l
