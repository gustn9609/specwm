true
# usage: WAITFILE=exp/Q81_DONE NEXT=scripts/q71c_stevo_base_mid.job.sh ./run.sh <gpu> scripts/q93_chain_after.job.sh tag — 마커가 생기면 다음 워커 실행
for i in $(seq 1 2000); do [ -f $SCRATCH/$WAITFILE ] && break; sleep 60; done
SHARD=0/1 bash $SCRATCH/$NEXT
