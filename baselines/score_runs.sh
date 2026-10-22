#!/bin/bash
# [10/17 frozen baselines] Score every clip of ONE runs/<model_arm>/ dir (<clip>/video.mp4 + timing.json) with the PROTOCOL rule-7 metrics and append
# the results to baselines/metrics/{vlm,vbench6,rpe,reaction}.json (keyed by the absolute video path, label = the arm dir name). Append-only: clips
# already scored under this label are skipped (filtered here AND by every scorer), so re-running after new clips arrive is cheap.
#   judge    scripts/vlm_clip_score.py            Qwen3-VL-32B on GPU 7 (:JUDGE_PORT), under flock /tmp/vlm_base.lock           -> metrics/vlm.json
#   vbench   scripts/vbench6.py --fps-norm 16     tools/vbench_env, under flock /tmp/gpu<G>.lock, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 -> metrics/vbench6.json
#   rpe      scripts/rpe_da3.py --program-from-path  DA3 vs baselines/programs/<P>.npz, under flock /tmp/gpu<G>.lock             -> metrics/rpe.json
#   reaction scripts/reaction_p5.py               CPU, only the *__P5* clips                                                      -> metrics/reaction.json
#   vlm_seg  scripts/vlm_seg_score.py             [10/19 EXP 1] only the *__P6* (60-s) clips: the judge on each 10-s piece (same prompts / server, under
#                                                 flock /tmp/vlm_base.lock) for drift-over-time plots                              -> metrics/vlm_seg.json
#   [10/19] rpe passes --seg-seconds 2: a 60-s P6 clip is aligned in 30 two-second segments (a 10-s clip keeps its 5; DA3 batch size unchanged)
# GPU work runs in batches of --batch clips with ONE flock hold per batch (keeps every hold short); _gpu_peak_run.sh runs inside the lock and logs
# the hold time and the peak GPU memory of our hold (nvidia-smi, whole GPU). Everything is logged to baselines/logs/score_<model_arm>.log.
# usage: /home/hk28965/specwm/baselines/score_runs.sh ARM_DIR GPU [--metrics DIR] [--batch N] [--skip judge,vbench,rpe,reaction] [--judge-port 30100] [--fps-norm 16] [--min-age S]
#   ARM_DIR = /home/hk28965/specwm/baselines/runs/<model_arm>      GPU = the GPU index assigned to this job (4 | 5 | 6)
#   --min-age S [10/17, additive, default 0 = every clip]: only clips whose timing.json exists and whose video.mp4 is older than S seconds (a clip still
#   being written by a generator must not be scored: the metrics files are append-only). gen/scorer.sh passes 120.
set -u
S=/home/hk28965/specwm; B=$S/baselines; R=$S/scripts/envrun.sh; VB=$S/tools/vbench_env/bin/python
ARM_DIR=${1:?usage: score_runs.sh ARM_DIR GPU}; G=${2:?usage: score_runs.sh ARM_DIR GPU}; shift 2
M=$B/metrics; BATCH=8; SKIP=""; JP=30100; FPSN=16; MINAGE=0
while [ $# -gt 0 ]; do case $1 in
  --metrics) M=$2; shift 2;; --batch) BATCH=$2; shift 2;; --skip) SKIP=$2; shift 2;; --judge-port) JP=$2; shift 2;; --fps-norm) FPSN=$2; shift 2;; --min-age) MINAGE=$2; shift 2;;
  *) echo "unknown option $1"; exit 1;; esac; done
ARM_DIR=$(cd "$ARM_DIR" && pwd -L) || { echo "no dir $ARM_DIR"; exit 1; }; LABEL=$(basename "$ARM_DIR"); mkdir -p "$M" $B/logs; LOG=$B/logs/score_$LABEL.log
VIDS=(); NOW=$(date +%s)
for v in $(ls -d "$ARM_DIR"/*/video.mp4 2>/dev/null | sort); do
  if [ "$MINAGE" -gt 0 ]; then [ -f "$(dirname "$v")/timing.json" ] || continue; [ $(( NOW - $(stat -c %Y "$v") )) -ge "$MINAGE" ] || continue; fi
  VIDS+=("$v"); done
[ ${#VIDS[@]} -gt 0 ] || { echo "no <clip>/video.mp4 under $ARM_DIR"; exit 1; }
ALLVIDS=("${VIDS[@]}")   # [10/19] step 4 narrows VIDS to the P5 clips; step 5 reads this copy
log() { echo "[$(date '+%m/%d %H:%M:%S')] $*" | tee -a "$LOG"; }
skip() { case ",$SKIP," in *,$1,*) return 0;; esac; return 1; }
todo() {   # $1 metrics json: print the videos not yet scored under this label (vbench6 keys by abspath, the others by the path given -> we always pass absolute paths)
  python3 - "$1" "$LABEL" "${VIDS[@]}" <<'EOF'
import json, os, sys
m, lab, vids = sys.argv[1], sys.argv[2], sys.argv[3:]
d = json.load(open(m)) if os.path.exists(m) else {}
for v in vids:
    r = d.get(v) or d.get(os.path.abspath(v)) or d.get(os.path.realpath(v))
    if not (isinstance(r, dict) and r.get("label") == lab and ("quality_score" in r or "rot" in r or "drift" in r or "switches" in r)): print(v)
EOF
}
run_batches() {   # $1 = step (judge | vbench | rpe | reaction), $2 = env assignments for the step ("" allowed), $3.. = command; LABEL=video specs appended per batch
  local step=$1 envs=$2; shift 2; local n=${#TODO[@]}; local i=0 t0 rc pk
  while [ $i -lt $n ]; do
    local specs=(); for v in "${TODO[@]:$i:$BATCH}"; do specs+=("$LABEL=$v"); done
    t0=$(date +%s); log "$step: batch $((i / BATCH + 1)) (${#specs[@]} clips)"
    case $step in
      judge|vlm_seg) flock /tmp/vlm_base.lock env JUDGE_PORT=$JP $envs "$@" "${specs[@]}" >> "$LOG" 2>&1; rc=$?; log "$step: batch done in $(( $(date +%s) - t0 )) s (rc $rc)";;
      reaction) env $envs "$@" "${specs[@]}" >> "$LOG" 2>&1; rc=$?; log "$step: batch done in $(( $(date +%s) - t0 )) s (rc $rc)";;
      *) pk=$(mktemp); flock /tmp/gpu$G.lock env CUDA_VISIBLE_DEVICES=$G $envs $B/_gpu_peak_run.sh $G "$pk" "$@" "${specs[@]}" >> "$LOG" 2>&1; rc=$?
         read -r held peak < "$pk" 2>/dev/null; rm -f "$pk"; local wall=$(( $(date +%s) - t0 ))
         log "$step: batch done, wall $wall s = lock wait $(( wall - ${held:-0} )) s + hold ${held:-?} s on gpu $G, peak GPU memory during our hold ${peak:-?} MiB (rc $rc)";;
    esac
    i=$((i + BATCH))
  done
}
log "score_runs $LABEL: ${#VIDS[@]} clips, gpu $G, metrics $M, batch $BATCH, skip '$SKIP'"
# 1. judge (GPU 7 server; CPU here)
if ! skip judge; then
  TODO=($(todo "$M/vlm.json")); log "judge: ${#TODO[@]} to do"
  [ ${#TODO[@]} -gt 0 ] && run_batches judge "" $R python $S/scripts/vlm_clip_score.py "$M/vlm.json"
fi
# 2. VBench 6 + MUSIQ drift (fps-normalised copies at $FPSN fps)
if ! skip vbench; then
  TODO=($(todo "$M/vbench6.json")); log "vbench: ${#TODO[@]} to do"
  [ ${#TODO[@]} -gt 0 ] && run_batches vbench "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=8" $VB $S/scripts/vbench6.py "$M/vbench6.json" --device cuda:0 --fps-norm $FPSN --tmp $B/logs/vbtmp_$LABEL
  rm -rf $B/logs/vbtmp_$LABEL
fi
# 3. RPE vs the commanded program (DA3)
if ! skip rpe; then
  TODO=($(todo "$M/rpe.json")); log "rpe: ${#TODO[@]} to do"
  [ ${#TODO[@]} -gt 0 ] && run_batches rpe "OMP_NUM_THREADS=8" $R python $S/scripts/rpe_da3.py "$M/rpe.json" --program-from-path --seg-seconds 2
fi
# 4. reaction on P5 (CPU)
if ! skip reaction; then
  P5=(); for v in "${VIDS[@]}"; do case $(basename "$(dirname "$v")") in *__P5*) P5+=("$v");; esac; done
  if [ ${#P5[@]} -gt 0 ]; then VIDS=("${P5[@]}"); TODO=($(todo "$M/reaction.json")); log "reaction: ${#P5[@]} P5 clips, ${#TODO[@]} to do"
    [ ${#TODO[@]} -gt 0 ] && run_batches reaction "OMP_NUM_THREADS=8" $R python $S/scripts/reaction_p5.py "$M/reaction.json"
  else log "reaction: no *__P5* clips in $LABEL"; fi
fi
# 5. [10/19 EXP 1] per-10-s-piece judge on the 60-s P6 clips (judge server; CPU here)
if ! skip vlm_seg; then
  P6=(); for v in "${ALLVIDS[@]}"; do case $(basename "$(dirname "$v")") in *__P6*) P6+=("$v");; esac; done
  if [ ${#P6[@]} -gt 0 ]; then VIDS=("${P6[@]}"); TODO=($(todo "$M/vlm_seg.json")); log "vlm_seg: ${#P6[@]} P6 clips, ${#TODO[@]} to do"
    [ ${#TODO[@]} -gt 0 ] && run_batches vlm_seg "" $R python $S/scripts/vlm_seg_score.py "$M/vlm_seg.json"
  fi
fi
python3 - "$M" "$LABEL" <<'EOF' | tee -a "$LOG"
import json, os, sys
m, lab = sys.argv[1], sys.argv[2]
for f, key in (("vlm", "quality"), ("vbench6", "quality_score"), ("rpe", "rot"), ("reaction", "mean_react_s")):
    p = os.path.join(m, f + ".json"); d = json.load(open(p)) if os.path.exists(p) else {}
    rs = [r for r in d.values() if isinstance(r, dict) and r.get("label") == lab and r.get(key) is not None]
    print("[score_runs] %-9s %s: %d clips scored%s" % (f, lab, len(rs), ("; mean %s %.3f" % (key, sum(r[key] for r in rs) / len(rs))) if rs else ""))
EOF
log "SCORE_DONE $LABEL"
