true
# Base v1 timing arms on S00 seed 9999 (161 frames): worldcache@480x832, bf16@360x512, worldcache@360x512. Q70 이 끝난 뒤 시작.
for i in $(seq 1 240); do [ -f $SCRATCH/exp/Q70_DONE ] && break; sleep 60; done
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out/timing_s00; CK=$PWD/lingbot-world-base-cam
run() { echo "########## BASE $1 ($(date)) ##########"; python -m specwm.bench_base_official --ckpt $CK --image $SCRATCH/lingbot-world/examples/00/image.jpg --action-path $SCRATCH/rt2_action_00 --prompt-file $SCRATCH/p_00.txt --frames 161 --seed 9999 --out $R/base_$1 "${@:2}"; }
run wc4_480x832 --max-area 399360 --worldcache 4
run bf16_360x512 --max-area 184320
run wc4_360x512 --max-area 184320 --worldcache 4
run wc8_480x832 --max-area 399360 --worldcache 8
touch $SCRATCH/exp/Q73_DONE; echo "[Q73_DONE] $(date)"
