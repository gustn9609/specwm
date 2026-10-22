true
# C1: Base 70-step 가 중간 해상도를 생성할 수 있는가 (S00 seed 9999, 161 f): 320x192 (61440), 240x136 (32640), 208x320 (73728), 160x112 (20480, 실패 예상). 이후 A3-Base: 35/20-step (+WorldCache official thr0.10 ret0.1) @360x512
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-base-cam
A="--image $SCRATCH/lingbot-world-v2/examples/00/image.jpg --action-path $SCRATCH/rt2_action_00 --prompt-file $SCRATCH/p_00.txt --frames 161 --seed 9999"
for AREA in 73728 61440 32640 20480; do echo "########## C1 BASE area=$AREA ($(date)) ##########"; python -m specwm.bench_base_official --ckpt $CK $A --max-area $AREA --out $R/lowres_sr/base70_$AREA; done
touch $SCRATCH/exp/Q95_DONE; echo "[Q95_DONE] $(date)"
echo "########## A3 BASE self-speculative stack @360x512 ($(date)) ##########"; T=$R/timing_s00
run() { [ -f $T/$1/video.mp4 ] && return; python -m specwm.bench_base_official --ckpt $CK $A --max-area 184320 --out $T/$1 "${@:2}"; }
run base_steps35_360x512 --steps 35
run base_steps35_wcoff_360x512 --steps 35 --worldcache 3 --wc-official --wc-thresh 0.10 --wc-ret 0.1
run base_steps20_360x512 --steps 20
run base_steps20_wcoff_360x512 --steps 20 --worldcache 3 --wc-official --wc-thresh 0.10 --wc-ret 0.1
touch $SCRATCH/exp/Q96_DONE; echo "[Q96_DONE] $(date)"
