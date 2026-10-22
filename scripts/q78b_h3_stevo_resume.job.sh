true
source $SCRATCH/h3w-venv/bin/activate; cd $SCRATCH/h3world/H3-World; source env.sh; export OMP_NUM_THREADS=8; R=$SCRATCH/lingbot-world-v2/specwm_out
export CPATH=$HOME/.local/include/python3.12:$HOME/.local/include/x86_64-linux-gnu/python3.12:$HOME/.local/include:$CPATH
python $SCRATCH/scripts/h3w_stevo.py --mode h3 --tasks $SCRATCH/data/stevo/tasks_video.json --out $R/stevo/h3_video --num-frames 243 --shard ${SHARD:-0/1}
touch $SCRATCH/exp/Q78_DONE; echo "[Q78B_DONE] $(date)"
