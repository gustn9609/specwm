true
# usage: WAITLOG=logs/q83_gN.*.out SHARD=i/n ./run.sh <gpu> scripts/q84b_h3world_after.job.sh tag — 해당 GPU 의 arm 재실행이 끝나면 H3-World 워커 추가
for i in $(seq 1 240); do grep -q "Q83_DONE" $SCRATCH/$WAITLOG 2>/dev/null && break; sleep 30; done
SHARD=$SHARD bash $SCRATCH/scripts/q84_h3world_shard.job.sh
