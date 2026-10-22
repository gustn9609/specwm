true
# v2 bid (v1 파이프라인, 경로에 'cam' 포함해야 control_type=cam) on S00 seed 9999: bf16, WorldCache-reimpl k4, WorldCache-official k3
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out/timing_s00; CK=$SCRATCH/lingbot-world-v2/bid_as_v1_cam
A="--image $SCRATCH/lingbot-world/examples/00/image.jpg --action-path $SCRATCH/rt2_action_00 --prompt-file $SCRATCH/p_00.txt --frames 161 --seed 9999 --max-area 399360"
run() { echo "########## V2 BID $1 ($(date)) ##########"; python -m specwm.bench_base_official --ckpt $CK $A --out $R/v2bid_$1 "${@:2}"; }
run bf16_480x832
run wc4_480x832 --worldcache 4
run wcoff3_480x832 --worldcache 3 --wc-official
touch $SCRATCH/exp/Q76B_DONE; echo "[Q76B_DONE] $(date)"
