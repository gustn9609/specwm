true
# usage: VARIANT=1p3b|pretrain|bid SHARD=i/n ./run.sh <gpu> scripts/q75_stevo_v2.job.sh q75_<variant>_i   (official generate.py, 1 GPU, 253 frames, 480x832)
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out/stevo/lingbot_v2_$VARIANT; mkdir -p $R
case $VARIANT in
  1p3b) ARGS="--task i2v-1.3B --ckpt_dir $PWD/lingbot-world-v2-1.3b-causal-fast --assets_dir $PWD/lingbot-world-v2-14b-causal-fast --infer_mode causal_fast --local_attn_size 18 --sink_size 6";;
  pretrain) echo "causal-pretrain checkpoint deleted 2026-09-17 (re-download robbyant/lingbot-world-v2-14b-causal-pretrain)"; ARGS="--task i2v-A14B --ckpt_dir $PWD/lingbot-world-v2-14b-causal-pretrain --assets_dir $PWD/lingbot-world-v2-14b-causal-fast --infer_mode causal_pretrain --local_attn_size 18 --sink_size 6 --t5_cpu --offload_model True";;
  *) echo "unknown VARIANT $VARIANT"; exit 1;;
esac
TASKS=${TASKS:-$SCRATCH/data/stevo/tasks_camera.json}; [ -n "$OUT" ] && R=$OUT && mkdir -p $R; FR=${FRAMES:-253}
SI=${SHARD%/*}; SN=${SHARD#*/}
python - <<PY > /tmp/stevo_v2_${VARIANT}_$SI.txt
import json; T=json.load(open("$TASKS"))
for i,t in enumerate(T):
    if i % $SN == $SI: print(t["id"], t["image"], t["action"])
PY
while read TID IMG ACT; do
  [ -f $R/$TID.mp4 ] && { echo "[skip] $TID"; continue; }
  PROMPT=$(python -c "import json;T=json.load(open('$TASKS'));print([t for t in T if t['id']=='$TID'][0]['camera_WM'])")
  echo "########## STEVO V2 $VARIANT $TID ($(date)) ##########"; T0=$(date +%s)
  python generate.py $ARGS --size 480*832 --frame_num $FR --image $IMG --action_path $ACT --prompt "$PROMPT" --base_seed 1234 --convert_model_dtype --save_file $R/$TID.mp4 2>&1 | grep -v "^\s*$" | tail -15
  echo "[TIME] $TID wall_s=$(( $(date +%s) - T0 ))"
done < /tmp/stevo_v2_${VARIANT}_$SI.txt
echo "[Q75_DONE $VARIANT shard $SHARD] $(date)"
