true
# MiniMax-H3 (DiffSynth, 1 GPU, no LoRA): S00 seed 9999 timing (480p, 720p) then StEvo video tasks (221, video_WM prompt). GPU 5 after q72 shard 2/3.
for i in $(seq 1 600); do grep -q "Q72_DONE shard 2/3" $SCRATCH/logs/q72_fast_2.*.out 2>/dev/null && break; sleep 60; done
source $SCRATCH/h3w-venv/bin/activate; cd $SCRATCH/h3world/H3-World; source env.sh; export OMP_NUM_THREADS=8; R=$SCRATCH/lingbot-world-v2/specwm_out
export CPATH=$HOME/.local/include/python3.12:$HOME/.local/include/x86_64-linux-gnu/python3.12:$HOME/.local/include:$CPATH   # inductor/flex_attention 컴파일에 Python.h 필요
echo "########## H3 S00 timing 832x480 ($(date)) ##########"
python $SCRATCH/scripts/h3w_stevo.py --mode h3 --tasks $SCRATCH/data/stevo/tasks_s00.json --out $R/timing_s00/h3_s00 --seed 9999 --num-frames 243
echo "########## H3 S00 timing 1280x720 ($(date)) ##########"
python $SCRATCH/scripts/h3w_stevo.py --mode h3 --tasks $SCRATCH/data/stevo/tasks_s00.json --out $R/timing_s00/h3_s00_720p --seed 9999 --num-frames 243 --height 720 --width 1280
echo "########## H3 StEvo video tasks ($(date)) ##########"
python $SCRATCH/scripts/h3w_stevo.py --mode h3 --tasks $SCRATCH/data/stevo/tasks_video.json --out $R/stevo/h3_video --num-frames 243
touch $SCRATCH/exp/Q78_DONE; echo "[Q78_DONE] $(date)"
