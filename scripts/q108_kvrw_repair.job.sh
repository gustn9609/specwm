true
# 재접지: target 은 있는데 kvrw_laundered 가 실패(848 vs 832 리사이즈 버그)한 task 들의 kvrw arm 재실행. GPU 5. 반복 실행 안전.
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; O=$R/stevo/reground; V2=$S/lingbot-world-v2; cd $V2; export PYTHONPATH=$V2
for i in $(seq 1 40); do
  n=0
  for TD in $O/*/target; do t=$(basename $(dirname $TD)); [ -f $TD/video.mp4 ] || continue; [ -f $O/$t/W18_sink6/kvrw_laundered_s1234/video.mp4 ] && continue; [ -f $O/$t.lock ] && continue
    n=$((n+1)); IMG=$(python -c "import json;T=json.load(open('$S/data/stevo/tasks_camera.json'));t=[x for x in T if x['id']=='$t'][0];print(t['image'])"); ACT=$(python -c "import json;T=json.load(open('$S/data/stevo/tasks_camera.json'));t=[x for x in T if x['id']=='$t'][0];print(t['action'])")
    echo "########## KVRW-REPAIR $t ($(date)) ##########"
    PYTHONPATH=$V2 python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast --action-root $ACT --prompt-file $O/$t.prompt.txt --scene 00 --image $IMG --frames 253 --seeds 1234 --base-video $R/stevo/lingbot_base/$t/video.mp4 --configs W18_sink6 --arms kvrw_laundered --kv-providers "laundered=$TD/video.mp4" --out $O/$t 2>&1 | grep -E "kvrw_laundered|실패|Traceback|Error"
  done
  [ $n -eq 0 ] && { [ $(ls $O/*/W18_sink6/kvrw_laundered_s1234/video.mp4 2>/dev/null | wc -l) -ge 85 ] && break; sleep 600; }
done
echo "[Q108_DONE] $(date)"
