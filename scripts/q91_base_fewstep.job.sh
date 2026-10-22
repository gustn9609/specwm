true
# Base few-step sweep on S00 seed 9999 (Great Wall, 480x832 budget, UniPC shift 10 CFG 5): 10 / 20 / 35 steps (70 = reference floor run)
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out/fewstep_s00; mkdir -p $R
for ST in 10 20 35; do echo "########## BASE steps=$ST ($(date)) ##########"
  python -m specwm.bench_base_official --ckpt $PWD/lingbot-world-base-cam --image $SCRATCH/lingbot-world-v2/examples/00/image.jpg --action-path $SCRATCH/rt2_action_00 --prompt-file $SCRATCH/p_00.txt --frames 161 --seed 9999 --max-area 399360 --steps $ST --out $R/base_steps$ST; done
touch $SCRATCH/exp/Q91_DONE; echo "[Q91_DONE] $(date)"
