true
# A3-Base self-speculative stack with step-index WorldCache (thr0.10 ret0.1): 35-step, 20-step @360x512; + stepidx thr0.06 ret0.2 @35-step
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out/timing_s00; CK=$PWD/lingbot-world-base-cam
A="--image $SCRATCH/lingbot-world-v2/examples/00/image.jpg --action-path $SCRATCH/rt2_action_00 --prompt-file $SCRATCH/p_00.txt --frames 161 --seed 9999 --max-area 184320"
run() { [ -f $R/$1/video.mp4 ] && return; echo "########## A3-STEPIDX $1 ($(date)) ##########"; python -m specwm.bench_base_official --ckpt $CK $A --out $R/$1 "${@:2}"; }
run base_steps35_wcstep_thr10_ret10_360x512 --steps 35 --worldcache 3 --wc-official --wc-stepidx --wc-thresh 0.10 --wc-ret 0.1
run base_steps20_wcstep_thr10_ret10_360x512 --steps 20 --worldcache 3 --wc-official --wc-stepidx --wc-thresh 0.10 --wc-ret 0.1
run base_steps35_wcstep_360x512 --steps 35 --worldcache 3 --wc-official --wc-stepidx
run base_wcoff3_stepidx_thr15_ret10_360x512 --worldcache 3 --wc-official --wc-stepidx --wc-thresh 0.15 --wc-ret 0.1
touch $SCRATCH/exp/Q106_DONE; echo "[Q106_DONE] $(date)"
