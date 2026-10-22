true
# H3-World 나머지 태스크 (GPU 4 를 hl35496 가 가져가 OOM 으로 중단된 35개) → 끝나면 재채점 트리거 → 이어서 Base 역순 워커 재개 (GPU 3)
source $SCRATCH/h3w-venv/bin/activate; cd $SCRATCH/h3world/H3-World; source env.sh; export OMP_NUM_THREADS=8; R=$SCRATCH/lingbot-world-v2/specwm_out
export CPATH=$HOME/.local/include/python3.12:$HOME/.local/include/x86_64-linux-gnu/python3.12:$HOME/.local/include:$CPATH
echo "########## H3-WORLD StEvo remaining ($(date)) ##########"
python $SCRATCH/scripts/h3w_stevo.py --mode h3world --tasks $SCRATCH/data/stevo/tasks_camera.json --out $R/stevo/h3world_lingbotcam --num-frames 243
rm -f $R/stevo/runs/h3world_lingbotcam/summary.json; touch $SCRATCH/exp/Q77_DONE; echo "[Q77B_DONE] $(date)"
echo "########## resume BASE reverse worker ($(date)) ##########"
SHARD=0/1 bash $SCRATCH/scripts/q71b_stevo_base_rev.job.sh
