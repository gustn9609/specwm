#!/bin/bash
# V-b: Base-independent camera-control metric — VGGT recovered path vs commanded poses (Umeyama-aligned pos err, fwd angle err) + rot-jump, for fast_alone / ours / ideal / Base on all StEvo camera tasks.
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; B=$R/verify_vggt; mkdir -p $B; source $S/b1-venv/bin/activate
python - <<'PY'
import json; t=json.load(open("/home/hk28965/specwm/data/stevo/tasks_camera.json")); open("/home/hk28965/specwm/specwm_out_web/_tasks.txt","w").write("\n".join("%s %s"%(x["id"],x["action"]) for x in t))
PY
while read tid act; do
  for kv in fast_alone=$R/stevo/reground/$tid/W18_sink6/fast_alone_s1234/video.mp4 ours=$R/stevo/reground/$tid/W18_sink6/kvrw_laundered_s1234/video.mp4 ideal=$R/stevo/reground/$tid/W18_sink6/resync_k6_s1234/video.mp4 base=$R/stevo/lingbot_base/$tid/video.mp4; do
    n=${kv%%=*}; v=${kv#*=}; [ -f $v ] || continue; [ -f $B/${tid}__$n.json ] && continue
    python $S/scripts/b1_vggt_pose.py --video $v --out $B/${tid}__$n.json --poses $act/poses.npy --max-frames 41 2>&1 | grep -E "Error|Traceback"
  done
done < $S/specwm_out_web/_tasks.txt
touch $S/exp/Q131_DONE; echo "[Q131_DONE] $(date)"
