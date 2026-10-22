#!/bin/bash
S=$SCRATCH; date "+지금 %H:%M"; squeue -u $USER -h -o "%.9i %.10j %.9T %.8M"; true
sacct -j 969209,969187,969210 -X -o JobID,JobName%9,Elapsed,State -n
echo "=== resync ==="; grep -E "^\[(load|base|fast_alone|ctrl|resync)|RESYNC_DONE|실패|Error" $S/resync.969209.out 2>/dev/null | tail -10; ls $S/lingbot-world-v2/specwm_out/resync_s05/*_error.json 2>/dev/null
echo "=== base steps ==="; grep -E "BASE official|gen_s|BASE_STEPS_DONE" $S/base_steps.969187.out 2>/dev/null | tail -8
echo "=== solar s1 ==="; grep -E "SOLAR_S1_DONE|solarwm: error" $S/solar_s1.969210.out 2>/dev/null | cut -c1-200; find $S/solar_out/stage1_s05 -name video.mp4 2>/dev/null | grep -v partial
