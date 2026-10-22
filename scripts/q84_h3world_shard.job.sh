true
source $SCRATCH/h3w-venv/bin/activate; cd $SCRATCH/h3world/H3-World; source env.sh; export OMP_NUM_THREADS=8; R=$SCRATCH/lingbot-world-v2/specwm_out
export CPATH=$HOME/.local/include/python3.12:$HOME/.local/include/x86_64-linux-gnu/python3.12:$HOME/.local/include:$CPATH
python $SCRATCH/scripts/h3w_stevo.py --mode h3world --tasks $SCRATCH/data/stevo/tasks_camera.json --out $R/stevo/h3world_lingbotcam --num-frames 243 --shard $SHARD
echo "[Q84_DONE shard $SHARD] $(date)"
