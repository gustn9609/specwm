true
for i in $(seq 1 240); do grep -q "Q83_DONE" /logs/q83_g4.2195406.out 2>/dev/null && break; sleep 30; done
ARMS="base_wc4_tau04_360x512:base:184320:--worldcache 4 --wc-tau0 0.04,base_wc4_gmax1_360x512:base:184320:--worldcache 4 --wc-gamma-max 1.0" bash /scripts/q83_arm.job.sh
SHARD=1/3 bash /scripts/q84_h3world_shard.job.sh
