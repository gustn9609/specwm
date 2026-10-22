true
# usage: SHARD=i/n ./run.sh <gpu> scripts/q71_stevo_base.job.sh q71_base_i   (Base v1 cam, 70-step UniPC, 253 frames; OOM 시 360x512 재시도)
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out/stevo/lingbot_base; CK=$PWD/lingbot-world-base-cam; mkdir -p $R
SI=${SHARD%/*}; SN=${SHARD#*/}
python - <<PY > /tmp/stevo_base_tasks_$SI.txt
import json; T=json.load(open("$SCRATCH/data/stevo/tasks_camera.json"))
L=list(enumerate(T)); L=L[3*len(L)//4:]+L[:3*len(L)//4]
for i,t in L:
    if i % $SN == $SI: print(t["id"], t["image"], t["action"])
PY
while read TID IMG ACT; do
  [ -f $R/$TID/video.mp4 ] && { echo "[skip] $TID"; continue; }; [ -f $R/$TID.lock ] && { echo "[locked] $TID"; continue; }; touch $R/$TID.lock
  echo "########## STEVO BASE $TID ($(date)) ##########"
  PF=$R/$TID.prompt.txt; python -c "import json;T=json.load(open('$SCRATCH/data/stevo/tasks_camera.json'));print([t for t in T if t['id']=='$TID'][0]['camera_WM'])" > $PF
  python -m specwm.bench_base_official --ckpt $CK --image $IMG --action-path $ACT --prompt-file $PF --frames 253 --seed 1234 --max-area 399360 --out $R/$TID \
    || { echo "[retry 360x512] $TID"; python -m specwm.bench_base_official --ckpt $CK --image $IMG --action-path $ACT --prompt-file $PF --frames 253 --seed 1234 --max-area 184320 --out $R/$TID; }
done < /tmp/stevo_base_tasks_$SI.txt
echo "[Q71_DONE shard $SHARD] $(date)"
