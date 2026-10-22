#!/bin/bash
# [10/16 frozen baselines] Waypoint-1.5-1B in ITS setting.
# ONE CLIP:   run.sh IMAGE PROGRAM_ID SEED OUTDIR [extra args for run_waypoint_program.py]
#   -> OUTDIR/actions.json + program_c2w.npy + program.json (encode_program.py: baselines/convert.py's program_to_c2w + encode_waypoint when importable,
#      else the equivalent inline encoder; K_yaw from baselines/calib.json [waypoint.K_yaw], else 118) and OUTDIR/video.mp4 (1024x512, 600 frames @ 60 fps
#      = 10 s) + OUTDIR/timing.json (PROTOCOL Section 4 fields) + run.log
# BATCH:      run.sh --batch LIST.tsv OUTROOT [extra args]      (10/17: one model load + one torch.compile + one warm-up for ALL clips)
#   LIST.tsv: one clip per line, tab-separated  IMAGE  PROGRAM_ID  SEED  [OUTNAME]   (# comments and blank lines ignored);
#   OUTNAME defaults to <set>__<scene>__<program> with set = the image's parent directory name and scene = its basename without extension, i.e. the
#   PROTOCOL Section 4 layout  OUTROOT/<set>__<scene>__<program>/{video.mp4, timing.json, actions.json, program_c2w.npy, program.json, run.log}.
#   Existing clips (video.mp4 + timing.json present) are skipped unless --overwrite is given; a failing clip writes OUTDIR/FAILED and the batch goes on;
#   OUTROOT/batch_jobs.json, batch.log, batch_summary.json record the session. Exit 3 if any clip failed.
# The caller selects the GPU and holds the lock, e.g.:
#   flock /tmp/gpu4.lock env CUDA_VISIBLE_DEVICES=4 /home/hk28965/specwm/baselines/waypoint/run.sh /home/hk28965/specwm/baselines/scenes/wp/highland_castle_loch.png P2 1234 /home/hk28965/specwm/baselines/runs/waypoint/wp__highland_castle_loch__P2
#   flock /tmp/gpu4.lock env CUDA_VISIBLE_DEVICES=4 /home/hk28965/specwm/baselines/waypoint/run.sh --batch /home/hk28965/specwm/baselines/waypoint/LIST.tsv /home/hk28965/specwm/baselines/runs/waypoint
# Non-interactive; ~2.5 min of compile warm-up (paid once per process) + ~20 s per clip on an RTX PRO 6000; peak VRAM ~7 GiB. Weights come from the local HF cache (offline).
set -e
S=/home/hk28965/specwm; B=$S/baselines/waypoint; R=$S/scripts/envrun.sh
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
if [ "$1" = "--batch" ]; then
  LIST=$2; OUTROOT=$3; shift 3
  [ -f "$LIST" ] && [ -n "$OUTROOT" ] || { echo "usage: run.sh --batch LIST.tsv OUTROOT [extra args]"; exit 1; }
  JOBS=${WP_JOBS:-$OUTROOT/batch_jobs.json}; mkdir -p "$OUTROOT" "$(dirname "$JOBS")"   # [10/17] WP_JOBS=path: per-batch jobs file (batch.log / batch_summary.json land next to it, so parallel batches never clobber OUTROOT/batch_jobs.json)
  $R python $B/batch_jobs.py "$LIST" "$OUTROOT" --jobs-out "$JOBS"          # encodes every clip's program (actions.json ...) and writes the jobs json
  $R python $B/run_waypoint_program.py --batch "$JOBS" "$@"
  exit $?
fi
IMG=$1; P=$2; SEED=$3; OUT=$4; shift 4
[ -n "$OUT" ] || { echo "usage: run.sh IMAGE PROGRAM_ID SEED OUTDIR  |  run.sh --batch LIST.tsv OUTROOT"; exit 1; }
mkdir -p "$OUT"
$R python $B/encode_program.py "$P" "$OUT"
$R python $B/run_waypoint_program.py --image "$IMG" --actions "$OUT/actions.json" --out "$OUT" --seed "$SEED" "$@"
