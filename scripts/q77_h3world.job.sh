true
# H3-World: S00 seed 9999 timing (rt2 poses -> keys) then StEvo camera tasks. GPU 4 after q72 shard 1/3.
true
source $SCRATCH/h3w-venv/bin/activate; cd $SCRATCH/h3world/H3-World; source env.sh; export OMP_NUM_THREADS=8; R=$SCRATCH/lingbot-world-v2/specwm_out
export CPATH=$HOME/.local/include/python3.12:$HOME/.local/include/x86_64-linux-gnu/python3.12:$HOME/.local/include:$CPATH   # inductor/flex_attention 컴파일에 Python.h 필요
echo "########## H3-WORLD S00 timing 832x480 ($(date)) ##########"
python $SCRATCH/scripts/h3w_stevo.py --mode h3world --tasks $SCRATCH/data/stevo/tasks_s00.json --out $R/timing_s00/h3world_s00 --pose-npy $SCRATCH/rt2_action_00/poses.npy --seed 9999 --num-frames 243
echo "########## H3-WORLD StEvo camera tasks ($(date)) ##########"
python $SCRATCH/scripts/h3w_stevo.py --mode h3world --tasks $SCRATCH/data/stevo/tasks_camera.json --out $R/stevo/h3world_lingbotcam --num-frames 243
touch $SCRATCH/exp/Q77_DONE; echo "[Q77_DONE] $(date)"
