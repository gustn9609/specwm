true
# Fast 14B quantization ablation on S00 seed 9999 (480x832 budget, 161 f): gptq6 / gptq6+kv4 / gptq6+kv4+tiled VAE (ours), plus fast_bf16 reference already in lowres_sr/fast14b_399360
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out/quant_s00; mkdir -p $R; CK=$PWD/lingbot-world-v2-14b-causal-fast
for ARM in gptq6 gptq6_kv4 ours; do echo "########## QUANT $ARM ($(date)) ##########"
  python -m specwm.stevo_gen --ckpt $CK --tasks $SCRATCH/data/stevo/tasks_s00.json --out $R/$ARM --arm $ARM --frames 161 --max-area 399360 --seed 9999 --t5-gpu; done
touch $SCRATCH/exp/Q90_DONE; echo "[Q90_DONE] $(date)"
