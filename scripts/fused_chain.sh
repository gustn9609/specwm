#!/bin/bash
# fused-commit probes: usage bash fused_chain.sh GPU ARM ARGS...
S=/home/hk28965/specwm; source $S/env.sh; cd $S/lingbot-world-v2
export OMP_NUM_THREADS=8 PYTHONPATH=$PWD:$S/tools/SageAttention
export CPATH=$HOME/.local/include/python3.12:$HOME/.local/include/x86_64-linux-gnu/python3.12:$HOME/.local/include:$CPATH
T=easy_lake,easy_greatwall,easy_stonehenge_sunset,easy_urban_rain,easy_dragon_night,pouring_water_into_cup,domino_falling,hourglass_accumulation,newtons_cradle,billiard_break,sand_filling_up_jar,sandcastle_wave_impact
G=$1; ARM=$2; shift 2
CUDA_VISIBLE_DEVICES=$G python -u -m specwm.res_split_infer --tasks $T --seeds ${SEEDS:-1234} --warm ${WARM:-1} --arm $ARM --out specwm_out/fused "$@" > specwm_out/fused/log_${ARM}_${SEEDS:-1234}.txt 2>&1; echo "DONE $ARM $?"
