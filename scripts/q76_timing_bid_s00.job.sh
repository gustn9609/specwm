true
# LingBot-World v2 bidirectional 14B (v1 파이프라인, 70-step UniPC shift10 CFG5 = Base 와 동일 설정 가정) on S00 seed 9999, Q73 뒤 GPU 2
for i in $(seq 1 600); do [ -f $SCRATCH/exp/Q73_DONE ] && break; sleep 60; done
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out/timing_s00; CK=$SCRATCH/lingbot-world-v2/bid_as_v1
echo "########## V2 BID 480x832 ($(date)) ##########"
python -m specwm.bench_base_official --ckpt $CK --image $SCRATCH/lingbot-world/examples/00/image.jpg --action-path $SCRATCH/rt2_action_00 --prompt-file $SCRATCH/p_00.txt --frames 161 --seed 9999 --max-area 399360 --out $R/v2bid_bf16_480x832
echo "########## V2 BID worldcache k4 480x832 ($(date)) ##########"
python -m specwm.bench_base_official --ckpt $CK --image $SCRATCH/lingbot-world/examples/00/image.jpg --action-path $SCRATCH/rt2_action_00 --prompt-file $SCRATCH/p_00.txt --frames 161 --seed 9999 --max-area 399360 --worldcache 4 --out $R/v2bid_wc4_480x832
touch $SCRATCH/exp/Q76_DONE; echo "[Q76_DONE] $(date)"
