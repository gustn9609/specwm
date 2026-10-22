true
# timing_s00 Base/bid arms 를 올바른 Great Wall 이미지(lingbot-world-v2/examples/00)로 재생성 (이전 실행은 dragon 중복 이미지 사용 → 시간 수치는 유효, 영상은 무효). 끝나면 H3-World 잔여 + Base 워커 재개.
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out/timing_s00; IMG=$SCRATCH/lingbot-world-v2/examples/00/image.jpg
A="--image $IMG --action-path $SCRATCH/rt2_action_00 --prompt-file $SCRATCH/p_00.txt --frames 161 --seed 9999"
run() { echo "########## RERUN $1 ($(date)) ##########"; python -m specwm.bench_base_official --ckpt $2 $A --out $R/$1 "${@:3}"; }
B=$PWD/lingbot-world-base-cam; BID=$SCRATCH/lingbot-world-v2/bid_as_v1_cam
run base_wc4_480x832 $B --max-area 399360 --worldcache 4
run base_bf16_360x512 $B --max-area 184320
run base_wc4_360x512 $B --max-area 184320 --worldcache 4
run base_wcoff3_360x512 $B --max-area 184320 --worldcache 3 --wc-official
run v2bid_bf16_480x832 $BID --max-area 399360
run v2bid_wc4_480x832 $BID --max-area 399360 --worldcache 4
run base_wc8_480x832 $B --max-area 399360 --worldcache 8
run v2bid_bf16_360x512 $BID --max-area 184320
run v2bid_wcoff3_360x512 $BID --max-area 184320 --worldcache 3 --wc-official
touch $SCRATCH/exp/Q82_DONE; echo "[Q82_DONE] $(date)"
bash $SCRATCH/scripts/q77b_h3world_rest.job.sh
