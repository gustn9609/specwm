#!/bin/bash
# GPU job queue with a disk guard: like gpu_jobs.sh, but a job starts only if /home has >= 3 GB free. usage: gpu_jobs2.sh JOBFILE
S=/home/hk28965/specwm; Q=$1
while [ -s $Q ]; do
  FREE=$(df --output=avail -k /home | tail -1); [ "$FREE" -lt 3000000 ] && { echo "$(date '+%m/%d %H:%M:%S') low disk ($((FREE/1024)) MB) - waiting"; sleep 120; continue; }
  G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1!=7 && $2<1000{print $1; exit}')
  [ -n "$G" ] || { sleep 10; continue; }
  L=$(head -1 $Q); tail -n +2 $Q > $Q.tmp && mv $Q.tmp $Q || { echo "queue file update failed"; sleep 60; continue; }
  C=${L//\{G\}/$G}; echo "$(date '+%m/%d %H:%M:%S') GPU $G: ${C:0:200}"; nohup bash -c "$C" > /dev/null 2>&1 &
  sleep 90
done
echo "$(date) queue empty"
