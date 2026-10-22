true
# LingBot v2 1.3B causal-fast & 14B causal-pretrain: S00 timing (161 frames) then StEvo camera tasks. GPU 3 after q74.
for i in $(seq 1 900); do [ -f $SCRATCH/exp/Q74_DONE ] && break; sleep 60; done
R=$SCRATCH/lingbot-world-v2/specwm_out
for V in 1p3b pretrain; do
  echo "########## V2 $V S00 timing ($(date)) ##########"
  VARIANT=$V SHARD=0/1 TASKS=$SCRATCH/data/stevo/tasks_s00.json OUT=$R/timing_s00/v2_${V}_s00 FRAMES=161 bash $SCRATCH/scripts/q75_stevo_v2.job.sh
done
echo "########## V2 1.3B StEvo ($(date)) ##########"; VARIANT=1p3b SHARD=0/1 bash $SCRATCH/scripts/q75_stevo_v2.job.sh
echo "########## V2 causal-pretrain StEvo ($(date)) ##########"; VARIANT=pretrain SHARD=0/1 bash $SCRATCH/scripts/q75_stevo_v2.job.sh
touch $SCRATCH/exp/Q79_DONE; echo "[Q79_DONE] $(date)"
