#!/bin/bash
# [baselines/sf1p3b] run several arms on ONE (image, program, seed) clip under one GPU lock hold (each generation process has its own hard
# timeout so the hold stays short). usage: arms.sh OUTDIR IMAGE PROGRAM GPU "ARM ARM ..." [SEED]
#   ARM in: published (tf, paper thresholds = arm A) | calibrated (tf, calib.json['sf1p3b']['calibrated'] = arm B) | nored (tf, TAU_RED=0 /
#   TAU_STAB=0.300 = sanity arm C: no step-1 accepts) | uniform4 (the 4-step reference on the same clip) | uniform2
# outputs: OUTDIR/<arm>/{video.mp4,timing.json,run.log,raw/,task/}, OUTDIR/arms.log
S=/home/hk28965/specwm; D=$S/baselines/sf1p3b; OUT=${1:?outdir}; IMG=${2:?image}; P=${3:?program}; G=${4:?gpu}; ARMS=${5:-"published calibrated nored uniform4"}; SEED=${6:-1234}
mkdir -p $OUT; OUT=$(cd $OUT && pwd); TO=${TIMEOUT:-300}
echo "[$(date +%F_%T)] waiting for /tmp/gpu$G.lock; arms: $ARMS; $IMG $P seed $SEED" >> $OUT/arms.log
flock /tmp/gpu$G.lock env CUDA_VISIBLE_DEVICES=$G bash -c "
  echo \"[\$(date +%F_%T)] lock acquired\" >> $OUT/arms.log; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i $G >> $OUT/arms.log
  for arm in $ARMS; do
    case \$arm in
      published)  env MODE=tf ARM=published TIMEOUT=$TO $D/run.sh $IMG $P $SEED $OUT/\$arm >> $OUT/arms.log 2>&1;;
      calibrated) env MODE=tf ARM=calibrated TIMEOUT=$TO $D/run.sh $IMG $P $SEED $OUT/\$arm >> $OUT/arms.log 2>&1;;
      nored)      env MODE=tf ARM=custom TAU_RED=0 TAU_STAB=0.300 TIMEOUT=$TO $D/run.sh $IMG $P $SEED $OUT/\$arm >> $OUT/arms.log 2>&1;;
      uniform4)   env MODE=uniform:4 TIMEOUT=$TO $D/run.sh $IMG $P $SEED $OUT/\$arm >> $OUT/arms.log 2>&1;;
      uniform2)   env MODE=uniform:2 TIMEOUT=$TO $D/run.sh $IMG $P $SEED $OUT/\$arm >> $OUT/arms.log 2>&1;;
      *) echo \"unknown arm \$arm\" >> $OUT/arms.log;;
    esac
    echo \"[\$(date +%F_%T)] \$arm done rc \$?\" >> $OUT/arms.log
  done
"
echo "[$(date +%F_%T)] lock released" >> $OUT/arms.log; echo ARMS_END >> $OUT/arms.log
