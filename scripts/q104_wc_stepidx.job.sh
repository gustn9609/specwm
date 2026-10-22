true
# WorldCache official port, step-index progress (공식과 동일 정의). Base 360x512 S00 seed 9999: thr0.06 ret0.2 / thr0.10 ret0.1 / thr0.10 ret0.1 @480x832. E1' 종료 후 GPU 4.
for i in $(seq 1 600); do [ -f $SCRATCH/exp/Q94_e1repro_DONE ] && break; sleep 60; done
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out/timing_s00; CK=$PWD/lingbot-world-base-cam
A="--image $SCRATCH/lingbot-world-v2/examples/00/image.jpg --action-path $SCRATCH/rt2_action_00 --prompt-file $SCRATCH/p_00.txt --frames 161 --seed 9999"
run() { [ -f $R/$1/video.mp4 ] && return; echo "########## WC-STEPIDX $1 ($(date)) ##########"; python -m specwm.bench_base_official --ckpt $CK $A --out $R/$1 "${@:2}"; }
run base_wcoff3_stepidx_360x512 --max-area 184320 --worldcache 3 --wc-official --wc-stepidx
run base_wcoff3_stepidx_thr10_ret10_360x512 --max-area 184320 --worldcache 3 --wc-official --wc-stepidx --wc-thresh 0.10 --wc-ret 0.1
run base_wcoff3_stepidx_thr10_ret10_480x832 --max-area 399360 --worldcache 3 --wc-official --wc-stepidx --wc-thresh 0.10 --wc-ret 0.1
touch $SCRATCH/exp/Q104_DONE; echo "[Q104_DONE] $(date)"
