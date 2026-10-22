true
# usage: SHARD=i/n ./run.sh <gpu> scripts/q72_stevo_fast.job.sh q72_fast_i
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-v2-14b-causal-fast
echo "########## STEVO FAST bf16 shard $SHARD ($(date)) ##########"
python -m specwm.stevo_gen --ckpt $CK --tasks $SCRATCH/data/stevo/tasks_camera.json --out $R/stevo/lingbot_fast --arm fast_bf16 --shard $SHARD --frames 253 --t5-gpu
echo "[Q72_DONE shard $SHARD] $(date)"
