#!/bin/bash
# [baselines/sf1p3b] Speculative Forcing in ITS setting: LingBot-World 2.0 1.3B causal-fast + the TRAINING-FREE verifier (arXiv 2610.22347 §3.3.1),
# 832x480, our 3-latent-frame chunks, sink 6 / window 18, 4-step schedule, seed fixed, one clip per (image, program).
# usage: run.sh IMAGE PROGRAM SEED OUTDIR
#   IMAGE   = scene image (png/jpg); PROGRAM = P1..P5 (baselines/programs/<P>.npz, else the PROTOCOL §2 table) or a dir with poses.npy [+intrinsics.npy]
#   OUTDIR/ : task/ (image, poses.npy, intrinsics.npy, task.json), raw/ (spec_early_tf output: loop.json, video.mp4), video.mp4, timing.json, run.log
# env overrides: MODE=tf|uniform:4|uniform:2 (default tf);
#   ARM=published|calibrated|custom (tf thresholds, default published):
#     published  = the paper's LingBot values tau_red 0.375 / tau_stab 0.300 (PROTOCOL rule 9, arm A; over-accepts on our setup, see ENV.md)
#     calibrated = the 10/17 re-calibration read from baselines/calib.json['sf1p3b']['calibrated'] (arm B: score quantile at the oracle acceptance
#                  rate rel_k <= 0.24 vs the 1.3B's own 4-step output, 3 neutral scenes x P1/P2, median over scenes; calib_1017/)
#     custom     = TAU_RED / TAU_STAB from the environment (e.g. TAU_RED=0 for the "no step-1 accepts" sanity arm C)
#   AREA=399360, FRAMES=161 (-> 153 generated frames with chunk 3), WARM=1 (compile warm-up clip, excluded from timing), STREAM=1 (side-stream
#   decode -> ttff/steady e2e fps), PROMPT="" (text; PROTOCOL rule 5: empty = primary), EXTRA="--tf-calib" (adds the untimed oracle calibration),
#   SAVE_LAT=0, TIMEOUT=seconds (kills the generation process after this long; 0 = none; envrun.sh exec's python, so `timeout` signals the python
#   process itself). The GPU comes from CUDA_VISIBLE_DEVICES set by the caller (shared GPU: flock /tmp/gpu<N>.lock).
# raw arm name: sf1p3b_tf_<ARM> for MODE=tf (custom -> sf1p3b_tf_custom<TAU_RED>_<TAU_STAB>), sf1p3b_uniform4 etc. otherwise.
set -e
S=/home/hk28965/specwm; D=$S/baselines/sf1p3b; R=$S/scripts/envrun.sh
IMG=$1; P=$2; SEED=$3; OUT=$4; [ -n "$OUT" ] || { echo "usage: run.sh IMAGE PROGRAM SEED OUTDIR"; exit 1; }
MODE=${MODE:-tf}; ARM=${ARM:-published}; AREA=${AREA:-399360}; FRAMES=${FRAMES:-161}; WARM=${WARM:-1}; STREAM=${STREAM:-1}; TIMEOUT=${TIMEOUT:-0}
case $ARM in
  published) TAU_RED=0.375; TAU_STAB=0.300;;
  calibrated) read TAU_RED TAU_STAB < <($R python -c "import json; c = json.load(open('$S/baselines/calib.json'))['sf1p3b']['calibrated']; print(c['tau_red'], c['tau_stab'])");;
  custom) TAU_RED=${TAU_RED:?custom arm needs TAU_RED}; TAU_STAB=${TAU_STAB:?custom arm needs TAU_STAB};;
  *) echo "unknown ARM $ARM (published|calibrated|custom)"; exit 1;;
esac
export SAVE_LAT=${SAVE_LAT:-0}; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
TID=$($R python $D/make_task.py "$IMG" "$P" "$OUT" --prompt "${PROMPT:-}" | tail -1)
if [ "$MODE" = tf ]; then ARMNAME=sf1p3b_tf_$ARM; [ $ARM = custom ] && ARMNAME=sf1p3b_tf_custom${TAU_RED}_${TAU_STAB}; else ARMNAME=sf1p3b_${MODE//:/}; fi
echo "[$(date +%F_%T)] $TID mode $MODE arm $ARM tau_red $TAU_RED tau_stab $TAU_STAB area $AREA frames $FRAMES seed $SEED gpu ${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "$OUT/run.log"
timeout -k 20 $TIMEOUT $R python -m specwm.spec_early_tf --tasks $TID --seeds $SEED --out "$OUT/raw" --tasks-json "$OUT/task/task.json" --prompt "${PROMPT:-}" --mode $MODE --model 1p3b \
   --area $AREA --frames $FRAMES --tau-red $TAU_RED --tau-stab $TAU_STAB --bucket 128 --warm $WARM --stream-decode $STREAM --arm $ARMNAME $EXTRA >> "$OUT/run.log" 2>&1
$R python $D/to_timing.py "$OUT/raw/$TID/${ARMNAME}_s$SEED" "$OUT" | tee -a "$OUT/run.log"
