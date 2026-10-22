true
# A2: LongLive-2.0-5B bf16 / fp8 @ 480p and 720p; LongLive-1.3B bf16 @ 480p. 10 s clips (5B: 24 fps -> 240 px frames ≈ 61 latent (8/block → 64); 1.3B: 16 fps -> 161 px ≈ 41 latent (3/block → 42))
source $SCRATCH/ll-venv/bin/activate; cd $SCRATCH/longlive/LongLive; export OMP_NUM_THREADS=8; export CPATH=$HOME/.local/include/python3.12:$HOME/.local/include/x86_64-linux-gnu/python3.12:$HOME/.local/include:$CPATH; export TRITON_CACHE_DIR=$SCRATCH/longlive/.triton; L=$(ls -d $SCRATCH/ll-venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib 2>/dev/null); export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$L:$LIBRARY_PATH; O=$SCRATCH/lingbot-world-v2/specwm_out/longlive; mkdir -p $O; export LLV2_TIME=1
B() { echo "########## LL $1 ($(date)) ##########"; python $SCRATCH/scripts/ll_bench.py --out $O/$1.json --save $O/$1.mp4 "${@:2}" 2>&1 | grep -v Warning | tail -25; }
B 5b_bf16_480p --config configs/inference.yaml --ckpt ../LongLive-2.0-5B/model_bf16.pt --frames 64 --height 480 --width 832
B 5b_bf16_720p --config configs/inference.yaml --ckpt ../LongLive-2.0-5B/model_bf16.pt --frames 64 --height 720 --width 1280
B 5b_fp8_480p --config configs/inference.yaml --ckpt ../LongLive-2.0-5B/model_bf16.pt --frames 64 --height 480 --width 832 --fp8
touch $SCRATCH/exp/Q99_DONE; echo "[Q99_DONE] $(date)"
