#!/bin/bash
# GPU6: 공식 Self-Forcing 1.3B 블록 지연 (block 1 vs 3), SDPA 폴백
true
source $SCRATCH/venv/bin/activate; cd $SCRATCH/selfforcing
for i in $(seq 1 120); do grep -q SF_DL_DONE $SCRATCH/logs/sf_download.log && break; sleep 30; done
for B in 3 1; do echo "########## SELF-FORCING block=$B ($(date)) ##########"; python sf_bench.py --block $B --frames 21 --out bench_block$B.json || echo "!!! FAILED block $B"; done
touch $SCRATCH/exp/Q51_DONE; echo "[Q51_DONE] $(date)"
