#!/bin/bash
# [10/16 frozen baselines; 10/17 batch mode] ABot-World-0-5B-LF in ITS setting: one clip per (scene image, control program, seed).
# usage (single clip, one model load per clip):
#   CUDA_VISIBLE_DEVICES=g /home/hk28965/specwm/baselines/abot/run.sh IMAGE PROGRAM_ID SEED OUTDIR [extra bench_abot.py args]
#   e.g. flock /tmp/gpu4.lock env CUDA_VISIBLE_DEVICES=4 /home/hk28965/specwm/baselines/abot/run.sh \
#          /home/hk28965/specwm/baselines/scenes/abot/desert_valley_3rd.png P2 1234 /home/hk28965/specwm/baselines/runs/abot/abot__desert_valley_3rd__P2
# usage (batch, ONE model load for all jobs — use this for the frozen generation):
#   python3 /home/hk28965/specwm/baselines/abot/make_jobs.py JOBS.json --out-root /home/hk28965/specwm/baselines/runs/abot --programs P1,P2,P3,P4,P5
#   flock /tmp/gpu4.lock env CUDA_VISIBLE_DEVICES=4 /home/hk28965/specwm/baselines/abot/run.sh --batch JOBS.json [extra bench_abot.py args]
#   -> every job's OUTDIR gets video.mp4 + timing.json + log.txt; the whole stdout goes to JOBS.log next to JOBS.json; exit 1 if any job failed.
#   extra: --offload-t5 on (official 19 GiB DynamicSwap path; default off = T5 resident) | --quant fp8-per-tensor (fallback; frozen arm = fp8-per-token)
#          | --blocks 11 --keep-frames 120 | --warmup-blocks 0
# caption = IMAGE's sibling .txt (same stem) if present, else empty (batch: the job's "prompt", written by make_jobs.py from the same rule); the web
# client's "| unknown |" prefix is always prepended by bench_abot.py.
# actions = per-block keys from PROGRAM_ID (baselines/abot/encode_abot.py): 11 blocks = 9 + 10 x 12 = 129 frames generated, first 120 (10.0 s @ 12 fps)
# written; bench_abot.py asserts that layout (block 0 = 9 frames) and resets the streaming VAE between streams (10/17 fix).
# out: OUTDIR/video.mp4 + OUTDIR/timing.json (PROTOCOL Section 4 fields) + OUTDIR/log.txt
set -o pipefail
S=/home/hk28965/specwm
if [ "$1" = "--batch" ]; then
  J=$2; shift 2; [ -f "$J" ] || { echo "usage: run.sh --batch JOBS.json [extra args]  (no such file: $J)"; exit 2; }
  LOG="${J%.json}.log"
  $S/tools/abot/run.sh python $S/baselines/abot/bench_abot.py --batch "$J" "$@" 2>&1 | tee "$LOG"; exit ${PIPESTATUS[0]}
fi
IMG=$1; P=$2; SEED=$3; OUT=$4; shift 4
[ -n "$IMG" ] && [ -n "$P" ] && [ -n "$SEED" ] && [ -n "$OUT" ] || { echo "usage: run.sh IMAGE PROGRAM_ID SEED OUTDIR [extra args]   |   run.sh --batch JOBS.json [extra args]"; exit 2; }
CAP=""; T="${IMG%.*}.txt"; [ -f "$T" ] && CAP="$(cat "$T")"
mkdir -p "$OUT"
$S/tools/abot/run.sh python $S/baselines/abot/bench_abot.py --image "$IMG" --program "$P" --seed "$SEED" --out "$OUT" --prompt "$CAP" "$@" 2>&1 | tee "$OUT/log.txt"; exit ${PIPESTATUS[0]}
