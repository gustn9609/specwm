true
# 공식 WorldCache 논리 arms @360x512 (480x832 는 캐시 메모리로 OOM): Base k=3, v2-bid k=3 — Q76B 뒤 GPU 2
for i in $(seq 1 900); do [ -f $SCRATCH/exp/Q76B_DONE ] && break; sleep 60; done
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out/timing_s00
A="--image $SCRATCH/lingbot-world/examples/00/image.jpg --action-path $SCRATCH/rt2_action_00 --prompt-file $SCRATCH/p_00.txt --frames 161 --seed 9999 --max-area 184320 --worldcache 3 --wc-official"
echo "########## BASE wc-official k3 360x512 ($(date)) ##########"; python -m specwm.bench_base_official --ckpt $PWD/lingbot-world-base-cam $A --out $R/base_wcoff3_360x512
echo "########## V2 BID wc-official k3 360x512 ($(date)) ##########"; python -m specwm.bench_base_official --ckpt $SCRATCH/lingbot-world-v2/bid_as_v1_cam $A --out $R/v2bid_wcoff3_360x512
echo "########## V2 BID bf16 360x512 ($(date)) ##########"; python -m specwm.bench_base_official --ckpt $SCRATCH/lingbot-world-v2/bid_as_v1_cam --image $SCRATCH/lingbot-world/examples/00/image.jpg --action-path $SCRATCH/rt2_action_00 --prompt-file $SCRATCH/p_00.txt --frames 161 --seed 9999 --max-area 184320 --out $R/v2bid_bf16_360x512
touch $SCRATCH/exp/Q73C_DONE; echo "[Q73C_DONE] $(date)"
