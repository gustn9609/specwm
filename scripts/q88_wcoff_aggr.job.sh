true
ARMS="base_wcoff3_thr10_360x512:base:184320:--worldcache 3 --wc-official --wc-thresh 0.10,base_wcoff3_thr15_360x512:base:184320:--worldcache 3 --wc-official --wc-thresh 0.15,base_wcoff3_thr10_ret10_360x512:base:184320:--worldcache 3 --wc-official --wc-thresh 0.10 --wc-ret 0.1,base_wcoff2_thr10_360x512:base:184320:--worldcache 2 --wc-official --wc-thresh 0.10" bash $SCRATCH/scripts/q83_arm.job.sh
SHARD=1/3 bash $SCRATCH/scripts/q84_h3world_shard.job.sh
