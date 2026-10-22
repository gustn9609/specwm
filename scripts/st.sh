#!/bin/bash
S=$SCRATCH
date "+지금 %H:%M"; squeue -u $USER -h -o "%.9i %.10j %.9T %.8M"
echo "=== vb_pre2 ==="; grep -E "downloaded|OK|실패|PRE_DONE|Error" $S/vb_pre2.log | tail -4
echo "=== stage1 dl ==="; du -sh $S/SolarWM-models/.cache 2>/dev/null; ls -la $S/SolarWM-models/SolarWM-5B-tf-stage1-81f/model.pt 2>/dev/null | awk '{print $5}'
echo "=== latency smoke ==="; grep -E "^\[fast_bf16\]|청크|LATENCY_DONE|실패|Error|Traceback" $S/lat_smoke.968298.out 2>/dev/null | tail -6
echo "=== solar s2 ==="; grep -vE "Warning|warn|^\s*$|Loading" $S/solar_s2.968325.out 2>/dev/null | tail -12
echo "=== base official ==="; grep -E "BASE official|BASE_OFFICIAL_DONE|Error|Traceback|gen_s|flicker" $S/base_official.968288.out 2>/dev/null | tail -4
