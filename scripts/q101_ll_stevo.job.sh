true
# LongLive-2.0-5B I2V on StEvo video tasks (221, 480x832, 64 latent = 253 px f @24fps ≈ 10.5 s). GPU 3 after q99.
for i in $(seq 1 600); do [ -f $SCRATCH/exp/Q99_DONE ] && grep -q "5b_fp8_480p" $SCRATCH/logs/q99_llbench.1688796.out && break; sleep 60; done
source $SCRATCH/ll-venv/bin/activate; cd $SCRATCH/longlive/LongLive; export OMP_NUM_THREADS=8; export CPATH=$HOME/.local/include/python3.12:$HOME/.local/include/x86_64-linux-gnu/python3.12:$HOME/.local/include:$CPATH; export TRITON_CACHE_DIR=$SCRATCH/longlive/.triton; export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
echo "########## LONGLIVE I2V StEvo ($(date)) ##########"; python inference.py --config_path configs/inference_i2v_stevo.yaml 2>&1 | grep -v Warning | grep -E "Error|Traceback|saved|Saving|/221|rank0" | tail -40
# rename rank0-<idx>-0_*.mp4 -> <task>.mp4
python - <<PY
import json, os, glob, shutil
S="$SCRATCH"; order=json.load(open(S+"/data/stevo/ll_i2v/order.json")); raw=S+"/lingbot-world-v2/specwm_out/stevo/longlive2_5b_raw"; out=S+"/lingbot-world-v2/specwm_out/stevo/longlive2_5b_video"; os.makedirs(out,exist_ok=True); n=0
for p in glob.glob(raw+"/rank0-*-0_*.mp4"):
    idx=int(os.path.basename(p).split("-")[1]); tid=os.path.splitext(order[idx])[0]; shutil.copy(p, out+"/"+tid+".mp4"); n+=1
print("renamed", n)
PY
touch $SCRATCH/exp/Q101_DONE; echo "[Q101_DONE] $(date)"
