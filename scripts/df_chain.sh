#!/bin/bash
# Draft-then-Finish A/B: waits for a free GPU, grabs it (80 GB reserve at import) and runs all five arms in one process
S=/home/hk28965/specwm; source $S/env.sh; cd $S/lingbot-world-v2
export OMP_NUM_THREADS=10 PYTHONPATH=$PWD:$S/tools/SageAttention
export CPATH=$HOME/.local/include/python3.12:$HOME/.local/include/x86_64-linux-gnu/python3.12:$HOME/.local/include:$CPATH
T=easy_lake,easy_greatwall,easy_stonehenge_sunset,easy_urban_rain,easy_dragon_night,pouring_water_into_cup,domino_falling,hourglass_accumulation,newtons_cradle,billiard_break,sand_filling_up_jar,sandcastle_wave_impact
O=specwm_out/df_ab
for try in $(seq 1 2000); do
  G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<1000{print $1; exit}')
  [ -n "$G" ] || { sleep 5; continue; }
  echo "$(date) try $try GPU $G"
  CUDA_VISIBLE_DEVICES=$G python -u -m specwm.draft_finish --tasks $T --seeds 1234 --configs fused:750,separate:750,fused:850,separate:850,fused:930 --out $O >> $O/log.txt 2>&1 && break
  grep -q "OutOfMemory" $O/log.txt && echo "OOM on $G, retry"; sleep 5
done
echo GEN_DONE
