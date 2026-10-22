#!/bin/bash
# generic GPU job queue: each line of JOBFILE is a shell command with {G} for the GPU index; a job starts on the first GPU with < 1 GB used
# (GPU 7 = VLM judge, excluded); waits 90 s after each launch so the new job's memory shows before the next check. usage: gpu_jobs.sh JOBFILE
S=/home/hk28965/specwm; Q=$1
while [ -s $Q ]; do
  G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1!=7 && $2<1000{print $1; exit}')
  [ -n "$G" ] || { sleep 10; continue; }
  L=$(head -1 $Q); sed -i 1d $Q; C=${L//\{G\}/$G}
  echo "$(date '+%m/%d %H:%M:%S') GPU $G: ${C:0:220}"; nohup bash -c "$C" > /dev/null 2>&1 &
  sleep 90
done
echo "$(date) queue empty"
