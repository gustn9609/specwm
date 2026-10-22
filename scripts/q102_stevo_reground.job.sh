true
# 재접지 arms on StEvo camera tasks (85): per task  fast_alone + resync_k6 (Base ctx 0-5, ideal) -> launder(v1 low expert) -> RePaint 16-step -> kvrw_laundered.  usage: SHARD=i/n ./run.sh <gpu> ...
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V1=$S/lingbot-world; V2=$S/lingbot-world-v2; O=$R/stevo/reground; mkdir -p $O; SI=${SHARD%/*}; SN=${SHARD#*/}
python - <<PY > /tmp/reground_tasks_$SI.txt
import json; T=json.load(open("$S/data/stevo/tasks_camera.json"))
for i,t in enumerate(T):
    if i % $SN == $SI: print(t["id"], t["image"], t["action"])
PY
while read TID IMG ACT; do
  BV=$R/stevo/lingbot_base/$TID/video.mp4; [ -f $BV ] || { echo "[skip] $TID no base"; continue; }
  [ -f $O/$TID/W18_sink6/kvrw_laundered_s1234/video.mp4 ] && { echo "[skip] $TID done"; continue; }
  [ -f $O/$TID.lock ] && { echo "[locked] $TID"; continue; }; touch $O/$TID.lock
  PF=$O/$TID.prompt.txt; python -c "import json;T=json.load(open('$S/data/stevo/tasks_camera.json'));print([t for t in T if t['id']=='$TID'][0]['camera_WM'])" > $PF
  echo "########## REGROUND $TID fast+resync ($(date)) ##########"; cd $V2; PYTHONPATH=$V2 python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast --action-root $ACT --prompt-file $PF --scene 00 --image $IMG --frames 253 --seeds 1234 --base-video $BV --configs W18_sink6 --arms fast_alone,resync_k6 --out $O/$TID
  FL=$O/$TID/W18_sink6/fast_alone_s1234/latent.pt; LD=$O/$TID/launder; TD=$O/$TID/target
  echo "########## REGROUND $TID launder ($(date)) ##########"; cd $V1; PYTHONPATH=$V1 python -m specwm.hybrid_refine --ckpt $V1/lingbot-world-base-cam --scene 00 --image $IMG --action-dir $ACT --prompt-file $PF --fast-latent $FL --frames 249 --schedules "0.40,0.20,0.08" --out $LD
  LV=$(ls -d $LD/REFINE_*/video.mp4 | head -1)
  echo "########## REGROUND $TID repaint ($(date)) ##########"; PYTHONPATH=$V1 python -m specwm.bench_base_prefix2 --ckpt $V1/lingbot-world-base-cam --image $IMG --action-path $ACT --prompt-file $PF --frames 253 --steps 16 --prefix-latents 9 --mode repaint --seed 1234 --prefix-video $LV --out $TD
  echo "########## REGROUND $TID kvrw ($(date)) ##########"; cd $V2; PYTHONPATH=$V2 python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast --action-root $ACT --prompt-file $PF --scene 00 --image $IMG --frames 253 --seeds 1234 --base-video $BV --configs W18_sink6 --arms kvrw_laundered --kv-providers "laundered=$TD/video.mp4" --out $O/$TID
  rm -f $O/$TID.lock
done < /tmp/reground_tasks_$SI.txt
echo "[Q102_DONE shard $SHARD] $(date)"
