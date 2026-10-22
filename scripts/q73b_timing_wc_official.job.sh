true
# 공식 WorldCache 논리(이식판) arms: Base k=3 @480x832, v2-bid k=3 @480x832 — Q74B 뒤 GPU 2
for i in $(seq 1 900); do [ -f $SCRATCH/exp/Q74B_DONE ] && break; sleep 60; done
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out/timing_s00
A="--image $SCRATCH/lingbot-world/examples/00/image.jpg --action-path $SCRATCH/rt2_action_00 --prompt-file $SCRATCH/p_00.txt --frames 161 --seed 9999 --max-area 399360 --worldcache 3 --wc-official"
echo "########## BASE wc-official k3 ($(date)) ##########"; python -m specwm.bench_base_official --ckpt $PWD/lingbot-world-base-cam $A --out $R/base_wcoff3_480x832
echo "########## V2 BID wc-official k3 ($(date)) ##########"; python -m specwm.bench_base_official --ckpt $SCRATCH/lingbot-world-v2/bid_as_v1 $A --out $R/v2bid_wcoff3_480x832
touch $SCRATCH/exp/Q73B_DONE; echo "[Q73B_DONE] $(date)"
