#!/bin/bash
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; L=$S/lingbot-world; date "+지금 %H:%M"; squeue -u $USER -h -o "%.9i %.6j %.9T %.8M %.14R"
sacct -j 972387,972416,972417 -X -o JobID,JobName%8,Elapsed,State -n
echo "== D targetcost =="; for d in $R/targetcost/area*; do [ -f $d/revisit.json ] && python3 -c "import json;r=json.load(open('$d/revisit.json'));print('$(basename $d)', r.get('hw'), r.get('steps'), r.get('gen_s'), r.get('peak_alloc_GiB'))"; done
echo "== E1 =="; f=$(ls -t $S/r4E1.*.out 2>/dev/null | head -1); grep -E "^\[W|실패|R4E1_DONE|Error" $f 2>/dev/null | cut -c1-130 | tail -16
echo "== E2 =="; f=$(ls -t $S/r4E2.*.out 2>/dev/null | head -1); grep -E "#####|BASE_PREFIX_DONE|R4E2_DONE|Error|Traceback" $f 2>/dev/null | cut -c1-120 | tail -8
echo "== repro 상태 =="; for sc in 05 01 02; do echo "S$sc fast=[$(ls $R/repro/S$sc/W18_sink6 2>/dev/null | grep fast | tr "\n" " ")] target=[$(for d in $R/repro/S$sc/target_s*; do [ -f $d/video.mp4 ] && basename $d; done | tr "\n" " ")]"; done
ls $R/resync55/W24_sink18/ 2>/dev/null | tr "\n" " "; echo
