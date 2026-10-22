true
for i in $(seq 1 600); do [ -f $SCRATCH/exp/Q99B_DONE ] && break; sleep 60; done
source $SCRATCH/ll-venv/bin/activate; cd $SCRATCH/longlive/LongLive; export OMP_NUM_THREADS=8; export CPATH=$HOME/.local/include/python3.12:$HOME/.local/include/x86_64-linux-gnu/python3.12:$HOME/.local/include:$CPATH; export TRITON_CACHE_DIR=$SCRATCH/longlive/.triton; export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
echo "########## LONGLIVE I2V StEvo missing 55 ($(date)) ##########"; python inference.py --config_path configs/inference_i2v_stevo_missing.yaml 2>&1 | grep -E "Error|Traceback" | tail -5
python - <<PY
import json, os, glob, shutil
S="$SCRATCH"; order=json.load(open(S+"/data/stevo/ll_i2v_missing/order.json")); raw=S+"/lingbot-world-v2/specwm_out/stevo/longlive2_5b_raw2"; out=S+"/lingbot-world-v2/specwm_out/stevo/longlive2_5b_video"; n=0
for p in glob.glob(raw+"/rank0-*-0_*.mp4"):
    idx=int(os.path.basename(p).split("-")[1]); tid=os.path.splitext(order[idx])[0]; shutil.copy(p, out+"/"+tid+".mp4"); n+=1
print("renamed", n, "total", len(glob.glob(out+"/*.mp4")))
PY
touch $SCRATCH/exp/Q101B_DONE; echo "[Q101B_DONE] $(date)"
