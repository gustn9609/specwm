true
ARMS="base_wc4_tau04_360x512:base:184320:--worldcache 4 --wc-tau0 0.04,base_wc4_gmax1_360x512:base:184320:--worldcache 4 --wc-gamma-max 1.0" bash $SCRATCH/scripts/q83_arm.job.sh
SHARD=1/3 bash $SCRATCH/scripts/q84_h3world_shard.job.sh
