#!/bin/bash
# Seam-fixed target at scale: RePaint 121 f, prefix 9, blend 6 latents (u=1) from the existing StEvo laundered videos -> KV-rewrite arm kvrw_blend6 on the 85 camera tasks. SHARD=i/n
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V1=$S/lingbot-world; V2=$S/lingbot-world-v2; CK=$V1/lingbot-world-base-cam; O=$R/stevo/reground; SI=${SHARD%/*}; SN=${SHARD#*/}
python - <<PY > /tmp/blend6_tasks_$SI.txt
import json; T=json.load(open("$S/data/stevo/tasks_camera.json"))
for i,t in enumerate(T):
    if i % $SN == $SI: print(t["id"], t["image"], t["action"])
PY
while read TID IMG ACT; do
  LV=$(ls -d $O/$TID/launder/REFINE_*/video.mp4 2>/dev/null | head -1); BV=$R/stevo/lingbot_base/$TID/video.mp4; PF=$O/$TID.prompt.txt; TD=$O/$TID/target_blend6
  [ -f "$LV" ] && [ -f $BV ] || { echo "[skip] $TID"; continue; }
  [ -f $O/$TID/W18_sink6/kvrw_blend6_s1234/video.mp4 ] && { echo "[done] $TID"; continue; }
  if [ ! -f $TD/video.mp4 ]; then echo "########## BLEND6 target $TID ($(date)) ##########"; cd $V1; PYTHONPATH=$V1 python -m specwm.bench_base_prefix2 --ckpt $CK --image $IMG --action-path $ACT --prompt-file $PF --frames 121 --steps 16 --prefix-latents 9 --blend-latents 6 --mode repaint --seed 1234 --prefix-video $LV --out $TD 2>&1 | grep -E "^\[|Error|Traceback"; fi
  echo "########## BLEND6 kvrw $TID ($(date)) ##########"; cd $V2; PYTHONPATH=$V2 python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast --action-root $ACT --prompt-file $PF --scene 00 --image $IMG --frames 253 --seeds 1234 --base-video $BV --configs W18_sink6 --arms kvrw_blend6 --kv-providers "blend6=$TD/video.mp4" --out $O/$TID 2>&1 | grep -E "^\[|Error|Traceback"
done < /tmp/blend6_tasks_$SI.txt
touch $S/exp/Q139_${SI}_DONE; echo "[Q139_DONE shard $SHARD] $(date)"
