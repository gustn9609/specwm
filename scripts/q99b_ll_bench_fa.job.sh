true
# A2 재측정 (flash-attn 2.8.3 부착, GPU 단독): 5B bf16 480p / +streaming VAE / 720p / fp8 480p / fp8+streaming. q101 뒤 GPU 3.
for i in $(seq 1 600); do [ -f $SCRATCH/exp/Q101_DONE ] && break; sleep 60; done
source $SCRATCH/ll-venv/bin/activate; cd $SCRATCH/longlive/LongLive; export OMP_NUM_THREADS=8; export CPATH=$HOME/.local/include/python3.12:$HOME/.local/include/x86_64-linux-gnu/python3.12:$HOME/.local/include:$CPATH; export TRITON_CACHE_DIR=$SCRATCH/longlive/.triton; export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH; O=$SCRATCH/lingbot-world-v2/specwm_out/longlive; export LLV2_TIME=1
B() { echo "########## LL-FA $1 ($(date)) ##########"; python $SCRATCH/scripts/ll_bench.py --out $O/$1.json --save $O/$1.mp4 "${@:2}" 2>&1 | grep -v Warning | grep -E "\[run|\[load\]|LLV2|Error|OutOfMemory|LL_BENCH" ; }
C="--config configs/inference.yaml --ckpt ../LongLive-2.0-5B/model_bf16.pt --frames 64"
B fa_5b_bf16_480p $C --height 480 --width 832
B fa_5b_bf16_480p_stream $C --height 480 --width 832 --streaming
B fa_5b_bf16_720p $C --height 720 --width 1280
B fa_5b_fp8_480p $C --height 480 --width 832 --fp8
B fa_5b_fp8_480p_stream $C --height 480 --width 832 --fp8 --streaming
touch $SCRATCH/exp/Q99B_DONE; echo "[Q99B_DONE] $(date)"
