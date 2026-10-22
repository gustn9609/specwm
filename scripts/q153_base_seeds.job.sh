#!/bin/bash
# Base 70-step rollouts (161 f, 480p) for the two failure tasks x seeds -> target batch for success-rate comparison
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD; CK=$PWD/lingbot-world-base-cam; O=$SCRATCH/lingbot-world-v2/specwm_out/spec_v0
for TS in $JOBS; do T=${TS%:*}; SD=${TS#*:}; D=$O/$T/base70_s$SD; [ -f $D/video.mp4 ] && continue
  IMG=$(python -c "import json;print([t for t in json.load(open('$SCRATCH/data/stevo/tasks_camera.json')) if t['id']=='$T'][0]['image'])"); ACT=$SCRATCH/data/stevo/actions/s-3_right-30_left-30
  echo "########## BASE $T s$SD ($(date)) ##########"; python -m specwm.bench_base_official --ckpt $CK --image $IMG --action-path $ACT --prompt-file $SCRATCH/exp/empty_prompt.txt --max-area 399360 --frames 161 --seed $SD --out $D 2>&1 | grep -E "^\[BASE|Error|Traceback"; done
touch $SCRATCH/exp/Q153_${SHARD}_DONE; echo "[Q153_DONE $SHARD] $(date)"
