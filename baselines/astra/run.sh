#!/bin/bash
# [10/16 frozen baselines] Astra (ICLR 2026, EternalEvan/Astra @ 3fb82e1, Wan2.1-T2V-1.3B based, 832x480 @ 20 fps) — official stack.
# single clip:  baselines/astra/run.sh IMAGE PROGRAM_ID SEED OUTDIR          -> OUTDIR/video.mp4 + timing.json (+ log.txt, camera.npz, sheet.jpg)
# batch (10/17): baselines/astra/run.sh batch JOBS.json [BATCH_LOG]           -> every job's OUTDIR/{video.mp4,timing.json,log.txt,camera.npz,sheet.jpg}
#   JOBS.json = [{"image": IMAGE, "program": PROGRAM_ID, "seed": 1234, "out": OUTDIR, "prompt": "..."}, ...]  (make_jobs.py writes one for a scene
#   set x programs). ONE model load (~11 s) and one warm-up block for the whole batch; a job with video.mp4 + timing.json is skipped (REDO=1 re-runs);
#   a job that fails writes OUTDIR/ERROR.txt and the batch goes on. "prompt" missing in a job -> the image's .txt sibling as in single mode.
#   PROGRAM_ID = P1..P5 (PROTOCOL §2 program, exact camera path via program_to_astra.py)  |  cam:N (Astra's own preset cam_type N = 1..7)
#   prompt     = $PROMPT if set (may be empty), else the IMAGE's .txt sibling (scenes/<set>/<scene>.txt), else "".
#   GPU        = the caller's CUDA_VISIBLE_DEVICES (and GPU lock):  flock /tmp/gpu5.lock env CUDA_VISIBLE_DEVICES=5 baselines/astra/run.sh ...
#   knobs (env): LATENTS (56 = 7 FULL blocks of 8; the clip is cut to FRAMES=201 = 10.05 s @ 20 fps. A partial last block collapses: the
#                10/16 P2 smoke with 50 latents = 6x8 + 2 produced garbage for exactly the 8 frames of the 2-latent block, see ENV.md),
#                BLOCK (8 latents per generation, their demo), STEPS (50, their default), FRAMES (201), WARM (1 = one discarded 8-latent
#                block before timing), T5_OFFLOAD (0 = T5 stays on the GPU like the official script), REDO (batch: 1 = regenerate existing)
set -o pipefail
S=/home/hk28965/specwm; B=$S/baselines/astra
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
X=""; [ "${T5_OFFLOAD:-0}" = 1 ] && X="--t5-offload"
[ -n "${U_SCALE:-}" ] && X="$X --u-scale $U_SCALE"     # re-calibration only; unset = frozen 0.15 (calib_astra.json)
COMMON="--latents ${LATENTS:-56} --frames ${FRAMES:-201} --block ${BLOCK:-8} --steps ${STEPS:-50} --warm ${WARM:-1} $X"
if [ "$1" = "batch" ]; then
  JOBS=$2; LOG=${3:-${JOBS%.*}.log}; [ -n "$JOBS" ] && [ -f "$JOBS" ] || { echo "usage: run.sh batch JOBS.json [BATCH_LOG]"; exit 1; }
  R=""; [ "${REDO:-0}" = 1 ] && R="--redo"
  # resolve missing prompts (image .txt sibling) into a private copy of the jobs file next to the log
  J2="${LOG%.*}.jobs_resolved.json"
  $S/tools/astra/venv/bin/python - "$JOBS" "$J2" <<'EOF'
import json, os, sys
jobs = json.load(open(sys.argv[1]))
for j in jobs:
    if "prompt" not in j:
        t = os.path.splitext(j["image"])[0] + ".txt"; j["prompt"] = open(t).read() if os.path.exists(t) else ""
json.dump(jobs, open(sys.argv[2], "w"), indent=1); print("batch: %d jobs -> %s" % (len(jobs), sys.argv[2]))
EOF
  $S/tools/astra/venv/bin/python $B/astra_gen.py --jobs "$J2" $R $COMMON 2>&1 | tee -a "$LOG"
  exit ${PIPESTATUS[0]}
fi
IMG=$1; PID=$2; SEED=${3:-1234}; OUT=$4
[ -n "$IMG" ] && [ -n "$PID" ] && [ -n "$OUT" ] || { echo "usage: run.sh IMAGE PROGRAM_ID SEED OUTDIR  |  run.sh batch JOBS.json [BATCH_LOG]"; exit 1; }
mkdir -p "$OUT"
if [ -z "${PROMPT+x}" ]; then T="${IMG%.*}.txt"; if [ -f "$T" ]; then PROMPT="$(cat "$T")"; else PROMPT=""; fi; fi
$S/tools/astra/venv/bin/python $B/astra_gen.py --image "$IMG" --program "$PID" --seed "$SEED" --out "$OUT" --prompt "$PROMPT" $COMMON 2>&1 | tee "$OUT/log.txt"
