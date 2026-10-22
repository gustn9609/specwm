true
source $SCRATCH/ll-venv/bin/activate; cd $SCRATCH/longlive/LongLive; mkdir -p wan_models/Matrix-Game-3.0
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["MG-LightVAE.pth","MG-LightVAE_v2.pth"]:
    try: hf_hub_download("Skywork/Matrix-Game-3.0", f, local_dir="wan_models/Matrix-Game-3.0"); print("DL_OK", f)
    except Exception as e: print("DL_FAIL", f, str(e)[:120])
PY
export OMP_NUM_THREADS=8; export CPATH=$HOME/.local/include/python3.12:$HOME/.local/include/x86_64-linux-gnu/python3.12:$HOME/.local/include:$CPATH; export TRITON_CACHE_DIR=$SCRATCH/longlive/.triton; export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH; O=$SCRATCH/lingbot-world-v2/specwm_out/longlive; export LLV2_TIME=1
B() { echo "########## LL-FA $1 ($(date)) ##########"; python $SCRATCH/scripts/ll_bench.py --out $O/$1.json --save $O/$1.mp4 "${@:2}" 2>&1 | grep -v Warning | grep -E "\[run|\[load\]|LLV2|Error|OutOfMemory|LL_BENCH" ; }
C="--config configs/inference.yaml --ckpt ../LongLive-2.0-5B/model_bf16.pt --frames 64"
B fa_5b_bf16_480p_lightvae_stream $C --height 480 --width 832 --streaming
B fa_5b_bf16_360p_lightvae_stream $C --height 352 --width 640 --streaming
touch $SCRATCH/exp/Q99D_DONE; echo "[Q99D_DONE] $(date)"
