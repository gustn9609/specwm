true
for i in $(seq 1 240); do grep -q "Q83_DONE" /logs/q83_g3.2195388.out 2>/dev/null && break; sleep 30; done
ARMS="v2bid_bf16_480x832:bid:399360:,base_wc8_480x832:base:399360:--worldcache 8" bash /scripts/q83_arm.job.sh
