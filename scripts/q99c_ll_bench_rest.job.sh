true
for i in $(seq 1 600); do [ -f $SCRATCH/exp/Q101B_DONE ] && break; sleep 60; done
source $SCRATCH/ll-venv/bin/activate; cd $SCRATCH/longlive/LongLive; export OMP_NUM_THREADS=8; export CPATH=$HOME/.local/include/python3.12:$HOME/.local/include/x86_64-linux-gnu/python3.12:$HOME/.local/include:$CPATH; export TRITON_CACHE_DIR=$SCRATCH/longlive/.triton; export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH; O=$SCRATCH/lingbot-world-v2/specwm_out/longlive; export LLV2_TIME=1
B() { echo "########## LL-FA $1 ($(date)) ##########"; python $SCRATCH/scripts/ll_bench.py --out $O/$1.json --save $O/$1.mp4 "${@:2}" 2>&1 | grep -v Warning | grep -E "\[run|\[load\]|LLV2|Error|OutOfMemory|LL_BENCH" ; }
C="--config configs/inference.yaml --ckpt ../LongLive-2.0-5B/model_bf16.pt --frames 64"
B fa_5b_bf16_480p_stream $C --height 480 --width 832 --streaming
B fa_5b_bf16_704p $C --height 704 --width 1280
B fa_5b_fp8_480p_stream $C --height 480 --width 832 --fp8 --streaming
B fa_5b_bf16_360p $C --height 352 --width 640
touch $SCRATCH/exp/Q99C_DONE; echo "[Q99C_DONE] $(date)"
