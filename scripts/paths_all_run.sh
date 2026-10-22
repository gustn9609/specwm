#!/bin/bash
# render the multi-baseline rollout-path figures once the SDVG path runs are complete (re-encoding on GPU 7 next to the judge)
S=/home/hk28965/specwm; O=$S/lingbot-world-v2/specwm_out; P6="sand_filling_up_jar sandcastle_wave_impact easy_lake easy_stonehenge_sunset easy_urban_rain pouring_water_into_cup"
until [ $(ls $O/sdvg_paths/*/ir_*_s*/video.mp4 2>/dev/null | wc -l) -ge 120 ] && [ $(ps aux | grep "[s]pec_ar_loop" | grep -c sdvg_paths) -eq 0 ]; do sleep 60; done
source $S/env.sh; cd $S
for sc in $P6; do CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/viz_paths_all.py --scene $sc 2>&1 | grep "\[FIG\]\|Error\|error" ; done
echo "PATHS_DONE $(date)"
