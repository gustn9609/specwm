#!/bin/bash
# [10/20] Resolution-split collaboration baseline (advisor's "strongest collaboration baseline": actions -> cloud, cloud generates at LOW resolution,
# sends it back, the device upsamples) inside the frozen protocol, as a NEW wrapper next to run_ours.sh (which is frozen and not edited).
# usage: run_ressplit.sh SET ARM GPU [CHUNKJSON]        (same CLI shape as run_ours.sh; the CALLER holds the lock:
#        flock /tmp/gpu$G.lock env CUDA_VISIBLE_DEVICES=$G /home/hk28965/specwm/baselines/gen/run_ressplit.sh SET ARM G CHUNK ; this script takes NO lock)
#   SET  = neutral_832 | neutral_832_nocap | astra   (832x480 only: area 399361; any other area is refused -- the cloud low resolution for the
#          640x352 / 1024x512 / 1280x704 sets is not defined yet)
#   ARM  = rs576  : cloud = 14B FP8 2-step (nodes 0,500) at 576x320 (--lo-area 193536), device = bilinear latent upsample to 832x480, no refiner
#          rs576r : the same cloud + device 1.3B refiner LoRA (env RS_LORA, default specwm_out/reslora/r850_930_pix/res_lora.pt: the r900 LoRA named in
#                   the 10/20 task was purged from spi0 and is not on lamarr either; r850_930_pix = ref_index [850, 930], rank 64, ffn, step 2784)
#   CHUNKJSON = optional: a plain task list OR a chunk json {"set","arm","tasks":[...]} (gen/make_queue_ressplit.py); default baselines/tasks/$SET.json
# engine: lingbot-world-v2/specwm/res_split_baseline.py (port of res_split_infer.py to the baseline task dicts; both parts serial on ONE GPU, streamed
#         832x480 decode for ttff / e2e; it writes the frozen layout itself in ours_to_timing's schema: chunk_s = cloud_chunk_s + device_chunk_s)
# out:  raw    baselines/raw/ours_<ARM>_<SET>/<task>/loop.json (+ spec.mp4 until converted)     log: baselines/logs/ours_<ARM>_<SET>.log, logs/run_ressplit.out
#       frozen baselines/runs/ours_<ARM>_<SET>/<task>/{video.mp4 (161 frames), timing.json}
# skip-if-done: tasks whose runs/ours_<ARM>_<SET>/<id>/timing.json exists are dropped (FORCE=1 re-runs); nothing left -> exit 0.
# env: RS_LO_AREA (default 193536 = exact 576x320 at the 832x480 aspect), RS_LORA (rs576r), FRAMES (default = the tasks' "frames", 165), FORCE, WARM (default 1)
# exit: the engine's rc when non-zero, else 2 when some task of the chunk has no frozen timing.json, else 0.
S=/home/hk28965/specwm; B=$S/baselines; R=$S/scripts/envrun.sh; cd $S
SET=$1; ARM=$2; G=$3; J=${4:-$B/tasks/$SET.json}
[ -n "$ARM" ] && [ -n "$G" ] || { echo "usage: run_ressplit.sh SET ARM GPU [CHUNKJSON]"; exit 1; }
[ -f "$J" ] || { echo "no tasks json $J"; exit 1; }
case "$ARM" in rs576|rs576r) ;; *) echo "unknown arm $ARM (rs576 | rs576r)"; exit 1;; esac
FORCE=${FORCE:-0}; WARM=${WARM:-1}; RS_LO_AREA=${RS_LO_AREA:-193536}
O=$B/raw/ours_${ARM}_${SET}; F=$B/runs/ours_${ARM}_${SET}; mkdir -p $O $F $B/logs; LOG=$B/logs/ours_${ARM}_${SET}.log
# task list + area (as run_ours.sh: a chunk json is copied as a plain list to $O/chunk_<name>; skip-if-done on the frozen dir)
read T A NALL NDONE JL TF <<< "$(python3 -I - "$J" "$F" "$FORCE" "$O" <<'EOF' 2>/dev/null | tail -1
import json, os, sys
t = json.load(open(sys.argv[1])); jl = sys.argv[1]
if isinstance(t, dict):
    t = t["tasks"]; jl = "%s/chunk_%s" % (sys.argv[4], os.path.basename(sys.argv[1]).replace(".gpu", "_gpu")); json.dump(t, open(jl, "w"), indent=1)
a = set(x["area"] for x in t); assert len(a) == 1, a
todo = [x["id"] for x in t if sys.argv[3] == "1" or not os.path.exists("%s/%s/timing.json" % (sys.argv[2], x["id"]))]
print(",".join(todo) or "-", a.pop(), len(t), len(t) - len(todo), jl, max(int(x.get("frames", 165)) for x in t))
EOF
)"
[ -n "$JL" ] || { echo "could not read task ids / area from $J"; exit 1; }
[ "$A" -eq 399361 ] || { echo "arm $ARM runs only at 832x480 (area 399361: sets neutral_832, neutral_832_nocap, astra); $SET has area $A -> refused" | tee -a $B/logs/run_ressplit.out; exit 1; }
FRAMES=${FRAMES:-${TF:-165}}; [ "$FRAMES" -le 165 ] || { echo "arm $ARM: long clips (frames $FRAMES) not supported"; exit 1; }
if [ "$T" = "-" ]; then echo "[$(date +%F_%H:%M)] ours $ARM $SET: nothing to do (all $NALL tasks of $J already frozen in $F)" | tee -a $B/logs/run_ressplit.out; exit 0; fi
NT=$(echo $T | tr ',' '\n' | wc -l); LORA=""
if [ $ARM = rs576r ]; then
  LORA=${RS_LORA:-$S/lingbot-world-v2/specwm_out/reslora/r850_930_pix/res_lora.pt}
  [ -f "$LORA" ] || { echo "arm rs576r: refiner LoRA $LORA missing (set RS_LORA)" | tee -a $B/logs/run_ressplit.out; exit 1; }
fi
export CUDA_VISIBLE_DEVICES=$G SAVE_LAT=${SAVE_LAT:-0} PATH=$HOME/.local/bin:$S/tools/ffmpeg-7.0.2-amd64-static:$PATH
echo "[$(date +%F_%H:%M)] ours $ARM $SET area $A lo_area $RS_LO_AREA frames $FRAMES gpu $G tasks $JL ($NT to run, $NDONE already frozen)${LORA:+ lora $LORA}" | tee -a $B/logs/run_ressplit.out | tee -a $LOG
$R python -m specwm.res_split_baseline --tasks-json $JL --tasks $T --arm $ARM --set $SET --out $O --runs $F --area $A --lo-area $RS_LO_AREA --frames $FRAMES --seed 1234 --warm $WARM --force $FORCE ${LORA:+--lora $LORA} >> $LOG 2>&1; RC=$?
MISS=0; for t in $(echo $T | tr ',' ' '); do [ -f $F/$t/timing.json ] || { MISS=$((MISS + 1)); echo "[missing] $F/$t/timing.json" | tee -a $LOG; }; done
echo "RS_DONE $ARM $SET rc $RC missing $MISS of $NT $(date +%F_%H:%M) -> $F" | tee -a $B/logs/run_ressplit.out | tee -a $LOG
[ $RC -ne 0 ] && exit $RC; [ $MISS -ne 0 ] && exit 2; exit 0
