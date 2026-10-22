#!/bin/bash
# [10/16 frozen baselines; 10/17 exact-resolution + timing update; 10/17 B chunk/queue update] Generate OUR arms on a frozen scene set at the baseline's
# resolution (PROTOCOL rule 4) — the only thing re-run when the algorithm changes.
# usage: run_ours.sh SET ARM GPU [TASKS_JSON]
#   SET  = wp | mg2 | abot | astra | neutral_wp | neutral_mg2 | neutral_abot | neutral_832 | neutral_832_nocap   (baselines/tasks/<SET>.json from make_tasks.py,
#          which pre-resizes every scene image to the baseline's exact size and stores area = W*H + 1: 1024x512 -> 524289, 640x352 -> 225281,
#          1280x704 -> 901121, 832x480 -> 399361; the +1 is needed because our loader's floor formula turns an exact 832x480 at area 399360 into 832x464)
#   ARM  = teacher | rt2 | solo3 | adapt25 | sf24 | student1p3b | sf1p3b_pub | sf1p3b_cal | adapt25c1hyb | rt2c1hyb | sf24c1hyb | rt2c1hyb32
#          | rt2c1hyb24 | mxW12 mxW9 mxK8 mxQ4 mx32 mx24 (+ c1hyb)   [10/19 thesis] see the MATRIX block below the arm comments
#          adapt25c1hyb [10/17 C] = OUR headline hybrid on a GPU PAIR (PROTOCOL rule 8): device = adapt25 (14B FP8, nodes 0,500,750, --use 0,1,
#          --adapt-budget 0.25 --adapt-node 2, gate-dir drafthead/rt2c1, --thr 0) shows every chunk at once and sends EVERY chunk to a cloud process
#          (14B FP8, --role cloud, --use 3 of nodes 0,250,500,750 = one step at t=750, --max-lag 99) that finishes it and returns it; the device
#          re-commits it into its KV window (specwm.two_machine, as specwm_out/twomach/run_adapthyb.sh). Pairs 0+1, 2+3, 4+5 (G = either member):
#          cloud on the LOWER index, device on the HIGHER, TCP port 32300 + lower index. Locks: BOTH /tmp/gpu<lo>.lock and /tmp/gpu<hi>.lock, in
#          ASCENDING order. A lock already held by an ANCESTOR process (the caller's flock, found via /proc/locks) is not re-taken (no self-dead-lock);
#          missing ones are taken by re-exec'ing this script under flock; caller holds only <hi> -> refused (exit 3: lock-order inversion).
#          Usual caller = a tier-4 chunk script baselines/gen/queue/4/NNN_<set>_adapt25c1hyb.sh (gen/make_queue_ours.py --tiers 4), which the worker
#          runs as `bash FILE G` WITHOUT a lock and which does `flock /tmp/gpu<lo>.lock flock /tmp/gpu<hi>.lock run_ours.sh SET adapt25c1hyb G CHUNK`.
#          Two frozen arms: runs/ours_adapt25c1hyb_dev_<SET> (the device-shown stream) and runs/ours_adapt25c1hyb_cloud_<SET> (the cloud-final stream);
#          raw/ours_adapt25c1hyb_<SET>/{cloud,device}; skip-if-done on the _dev frozen dir; no --timed-decode (two_machine records it for solo only:
#          ttff / e2e / decode are null); logs logs/ours_adapt25c1hyb_<SET>_{cloud,device}.log; the hold time + nvidia-smi peak MiB of both GPUs are
#          logged as HYB_HOLD in logs/ours_adapt25c1hyb_<SET>.log. The cloud is killed when the device fails; both are always cleaned up on exit.
#          rt2c1hyb / sf24c1hyb [10/19 EXP 2 / EXP 3] = more hybrids with the SAME pair mechanics, cloud flags, two frozen streams
#          (runs/ours_<ARM>_dev_<SET>, runs/ours_<ARM>_cloud_<SET>), logs logs/ours_<ARM>_<SET>{,_cloud,_device}.log:
#            rt2c1hyb  = light device: rt2 (nodes 0,500 --use 0,1, --thr 0; = specwm_out/twomach/run_832h.sh / run_43j.sh rt2c1hyb), cloud 1 step at t=750 + KV feedback
#            sf24c1hyb = device runs Speculative Forcing on the 14B exactly as the sf24 arm (token-wise early exit, head14b_nob43/head.pt, eps 0.24, bucket 128,
#                        nodes 0,250,500,750 --use 0,1,2,3, dense clean commit = --commit separate; two_machine --ee-head), cloud 1 step at t=750 + KV feedback
#            rt2c1hyb32 [10/19 EXP 2 light device] = rt2c1hyb with a 32-GiB device: the DEVICE process adds --window 12 --kv fp8 --cap 32 --lowmem 1
#                        (two_machine: FP8 KV ring buffer, allocator capped at 32 GiB from process start = an OOM is a real "does not fit", block-wise FP8
#                        load so the bf16 checkpoint never sits on the GPU, old KV dropped before prepare, spec.mp4 decoded frame by frame with the KV
#                        resident; memory records in its timing.json "mem"; [10/19 PM] --vae-dtype bf16 --vae-tiles 2,2: the device's spec.mp4 is decoded by a
#                        bf16 copy of the VAE in 2x2 spatial tiles, so 832x480 / 1024x512 fit the cap; ~51 dB vs the fp32 decode, PROTOCOL §7); the cloud is EXACTLY rt2c1hyb's (window $W, bf16 KV, no cap). Frozen streams
#                        runs/ours_rt2c1hyb32_{dev,cloud}_<SET>; the dev conversion records kv_window 12. Queued only on the sets whose resolution fits
#                        (tier 7, gen/make_queue_light.py; PROTOCOL §7).
#          [10/19 thesis] rt2c1hyb24 = rt2c1hyb32 with a 24-GiB device: the DEVICE adds --window 12 --kv fp8 --cap 24 --lowmem 1 --vae-dtype bf16 --vae-tiles 2,2
#                        --wq gptq4 --wq-file specwm_out/gptq4/14b_g128_832.pt (two_machine: packed INT4 GPTQ-4 g128 block linears from the OFFLINE file written by
#                        specwm.gptq_save, loaded block by block on the CPU -> GPU, Triton dequant when available; the bf16 checkpoint never on the GPU); the device
#                        conversion records wq gptq4 (WQD). The cloud is unchanged (--wq fp8 from HX; two_machine argparse keeps the LAST --wq / --window, so XD wins).
#          MATRIX [10/19 thesis] mx<X> (device only = two_machine --role solo on the rt2 schedule nodes 0,500 --use 0,1) and mx<X>c1hyb (the same device flags + the standard
#                        cloud of rt2c1hyb: nodes 0,250,500,750 --use 3 --max-lag 99, --thr 0 on the device) = ONE squeeze at a time on top of rt2, then the two bundles:
#                        mxW12 --window 12 | mxW9 --window 9 | mxK8 --kv fp8 (window 18) | mxQ4 --wq gptq4 --wq-file F --lowmem 1 (window 18) | mx32 = the rt2c1hyb32 device
#                        flags | mx24 = the rt2c1hyb24 device flags (table mx_flags below: name -> "device flags|window for the record|device wq"). Solo mx arms skip the
#                        pair mechanics (ordinary single-GPU run, frozen runs/ours_mx<X>_<SET>); the capped solo arms (mx32, mx24) run with --timed-decode 0 (a second
#                        VAE on a side stream is not what a capped device holds; ttff / e2e / decode_s null, said in "nulls"). Hybrids: runs/ours_mx<X>c1hyb_{dev,cloud}_<SET>.
#          sf1p3b_pub / sf1p3b_cal = Speculative Forcing in ITS setting (PROTOCOL rule 9): LingBot 1.3B causal-fast + the training-free verifier
#          (specwm/spec_early_tf.py --mode tf --model 1p3b --stream-decode 1) at the published thresholds (calib.json['sf1p3b']['published'] 0.375/0.300)
#          or the 10/17 re-calibrated ones (calib.json['sf1p3b']['calibrated'] 0.2919/0.3112); 832x480 ONLY (its own setting: sets astra, neutral_832,
#          neutral_832_nocap) — any other area is refused.
#   GPU  = CUDA device index; the CALLER holds the lock:  flock /tmp/gpu$G.lock env CUDA_VISIBLE_DEVICES=$G /home/hk28965/specwm/baselines/run_ours.sh SET ARM G
#          (this script takes NO lock itself: nesting one on the same file would dead-lock; baselines/gen/worker.sh is the usual caller)
#   TASKS_JSON = optional override of the tasks file: a plain task list OR a chunk json {"set","arm","tasks":[...]} from baselines/gen/make_queue.py
#          (e.g. a temporary single-entry json for a smoke test); SET still names the output dirs.
# skip-if-done: task ids whose frozen result runs/ours_<ARM>_<SET>/<id>/timing.json exists are dropped (FORCE=1 re-runs them); nothing left -> exit 0.
# [10/19 audit] env RUNS_SUFFIX (default empty = unchanged): every output dir and log gets it -> raw/ours_<ARM>_<SET>$RUNS_SUFFIX, runs/ours_<ARM>_<SET>$RUNS_SUFFIX
#       (hybrids: runs/ours_<ARM>_{dev,cloud}_<SET>$RUNS_SUFFIX), logs/ours_<ARM>_<SET>$RUNS_SUFFIX{,_cloud,_device}.log; skip-if-done looks at the SUFFIXED runs dir.
#       _idle = timing re-measure on an exclusively held GPU; the frozen videos/metrics stay in the un-suffixed dir (gen/queue/8/301; gen/idle_timing_map.json
#       tells report.py which dir to prefer for the timing columns). _dit = sf24 check with STREAM_DECODE=0 (gen/queue/8/302).
#       env STREAM_DECODE (default 1 = unchanged): the spec_early arms' --stream-decode flag (sf24, student1p3b, sf1p3b_*); 0 = no side-stream decode at all,
#       so the chunk timer holds only DiT work (ttff_s / e2e_s / steady_fps_e2e become null, e2e_s falls back to prepare + gen + full decode; ours_to_timing "nulls").
# [10/19 EXP 1] long clips: FRAMES defaults to the tasks' own "frames" (165 for every 10-s set; 969 for the P6 sets <set>_long = 243 latents = 81 chunks); FRAMES > 165
#       adds --long-clip 1 (two_machine / spec_early: KV caches dropped before prepare, frame-by-frame final decode to the host) and, for the spec_early arms,
#       --warm-frames 165 (the dynamic-shape warm clip stays 10 s); ours_to_timing.py trims P6 clips to 961 frames (60.0 s). sf1p3b_* arms refuse long sets.
# clip: --frames 165 = 42 latent frames = 14 chunks of 3 (165 generated frames); baselines/ours_to_timing.py trims the video to 161 frames (10.0 s, crf 18)
#       and writes the PROTOCOL §4 timing.json (timing over all 14 chunks). two_machine arms run with --timed-decode $TIMED_DECODE (default 1: side-stream
#       decode of every committed chunk -> ttff_s / e2e_s / steady_fps_e2e / decode_s), spec_early arms with --stream-decode 1. SAVE_LAT=0: the
#       spec_early arms write no raw lat.pt (two_machine always writes one; it is purged after the conversion).
# KV window: 18 (our standard); 12 above ~1280x704 for the two_machine arms (window 18 OOMs at 95 GB, facts §42); the spec_early arms have a fixed
#       sink 6 / window 18 (not configurable in spec_early.py / spec_early_tf.py), recorded as such.
# 1280x704 smoke (10/17 B, GPU 4, neutral_abot__easy_lake__P2, teacher, window 12, --timed-decode 1): no OOM, torch peak alloc 84.78 / reserved 84.82 GiB
#       (timing.json), whole-process 91.8 GiB by nvidia-smi (93995 MiB of 97887; CUDA context + compile caches on top of the torch pool), 98 s per clip
#       (steady 2.01 fps gen / 1.61 e2e, TTFF 18.2 s), lock hold 515 s for load + warm-up + 1 clip. --timed-decode stays 1 at every area (torch peak
#       < 90 GiB); if a 1280x704 chunk ever OOMs run it with TIMED_DECODE=0 (ttff_s / e2e_s / steady_fps_e2e / decode_s become null, said in "nulls").
#       sf24 at 1280x704 (spec_early, fixed window 18, FP8 14B) is untested and the most likely OOM candidate (facts §42).
# out: raw   baselines/raw/ours_<ARM>_<SET>/<task>/{timing.json [+ lat.pt, spec.mp4 until purged]} + summary.json (two_machine)
#            or <task>/<arm>_s1234/{loop.json [+ video.mp4 until purged]} (spec_early); after a successful conversion lat.pt / spec.mp4 / video.mp4 are deleted
#      frozen baselines/runs/ours_<ARM>_<SET>/<task>/{video.mp4 (161 frames), timing.json}      log: baselines/logs/ours_<ARM>_<SET>.log, run_ours.out
# exit status: the generation's rc when it failed, else 2 when some task of the chunk has no frozen timing.json after the conversion, else 0.
S=/home/hk28965/specwm; B=$S/baselines; R=$S/scripts/envrun.sh; cd $S
SET=$1; ARM=$2; G=$3; J=${4:-$B/tasks/$SET.json}
[ -n "$ARM" ] && [ -n "$G" ] || { echo "usage: run_ours.sh SET ARM GPU [TASKS_JSON]"; exit 1; }
[ -f "$J" ] || { echo "no tasks json $J (run make_tasks.py)"; exit 1; }
IS_HYB=0; case "$ARM" in adapt25c1hyb|rt2c1hyb|sf24c1hyb|rt2c1hyb32|rt2c1hyb24|mx*c1hyb|rt2c1hyb32g50|rt2c1hyb32g25|rt2c1hyb32g12|rt2c1hyb32g0|rt2c1hyb32o50|rt2c1hyb32o25|rt2c1hyb32o12|rt2c1hyb32d|rt2c1hyb32L1) IS_HYB=1;; esac   # [10/19] every hybrid arm uses the pair mechanics below; [10/19 thesis] + rt2c1hyb24, mx<X>c1hyb
if [ $IS_HYB = 1 ]; then   # [10/17 C] GPU-pair locks, ascending, never re-taking one an ancestor holds
  case "$G" in 0|1|2|3|4|5) ;; *) echo "$ARM: GPU $G has no pair (pairs 0+1, 2+3, 4+5)"; exit 1;; esac
  PG=$((G ^ 1)); LO=$(( G < PG ? G : PG )); HI=$(( G < PG ? PG : G ))
  anc_holds() {   # 1 when an ancestor of this shell (or the shell itself) holds the flock on file $1 (/proc/locks: pid + maj:min:inode)
    local f=$1 ino hold p h; [ -e "$f" ] || { echo 0; return; }; ino=$(stat -c %i "$f")
    hold=$(awk -v i="$ino" '$2 == "FLOCK" { n = split($6, a, ":"); if (a[n] == i) print $5 }' /proc/locks)
    p=$$; while [ -n "$p" ] && [ "$p" -gt 1 ] 2>/dev/null; do
      for h in $hold; do [ "$h" = "$p" ] && { echo 1; return; }; done
      p=$(awk '/^PPid:/ {print $2}' /proc/$p/status 2>/dev/null); done; echo 0; }
  HL=$(anc_holds /tmp/gpu$LO.lock); HH=$(anc_holds /tmp/gpu$HI.lock)
  if [ "$HL" = 1 ] && [ "$HH" = 1 ]; then :
  elif [ "$HH" = 1 ]; then echo "$ARM: the caller holds /tmp/gpu$HI.lock but not /tmp/gpu$LO.lock; taking gpu$LO now would invert the lock order (dead-lock risk) -> refused; take gpu$LO first (or no lock)"; exit 3
  elif [ "$HL" = 1 ]; then exec flock /tmp/gpu$HI.lock "$B/run_ours.sh" "$@"
  else exec flock /tmp/gpu$LO.lock flock /tmp/gpu$HI.lock "$B/run_ours.sh" "$@"; fi
fi
FRAMES=${FRAMES:-}; FORCE=${FORCE:-0}; TIMED_DECODE=${TIMED_DECODE:-1}   # [10/19] FRAMES default = the tasks' "frames" (165 for every 10-s set), read below
RUNS_SUFFIX=${RUNS_SUFFIX:-}; STREAM_DECODE=${STREAM_DECODE:-1}   # [10/19 audit] RUNS_SUFFIX: raw/runs/log dirs get it (e.g. _idle, _dit); STREAM_DECODE: spec_early arms' --stream-decode (default 1 = unchanged)
O=$B/raw/ours_${ARM}_${SET}${RUNS_SUFFIX}; F=$B/runs/ours_${ARM}_${SET}${RUNS_SUFFIX}; [ $IS_HYB = 1 ] && { F=$B/runs/ours_${ARM}_dev_${SET}${RUNS_SUFFIX}; FC=$B/runs/ours_${ARM}_cloud_${SET}${RUNS_SUFFIX}; mkdir -p $FC; }; mkdir -p $O $F $B/logs
# task list + area; a chunk json ({"tasks": [...]}) is copied as a plain list to $O/chunk_<name> because the loaders (SPECWM_TASKS / --tasks-json) iterate a list
read T A NALL NDONE JL TF <<< "$($R python - "$J" "$F" "$FORCE" "$O" <<'EOF' 2>/dev/null | tail -1
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
[ -n "$FRAMES" ] || FRAMES=${TF:-165}; LONG=""; LONGSE=""   # [10/19 EXP 1] long clips (P6 sets: 969)
if [ "$FRAMES" -gt 165 ]; then LONG="--long-clip 1"; LONGSE="--long-clip 1 --warm-frames 165"; case $ARM in sf1p3b_*) echo "arm $ARM: long clips (frames $FRAMES) not supported"; exit 1;; esac; fi
J=$JL
if [ "$T" = "-" ]; then echo "[$(date +%F_%H:%M)] ours $ARM $SET: nothing to do (all $NALL tasks of $J already frozen in $F)" | tee -a $B/logs/run_ours.out; exit 0; fi
NT=$(echo $T | tr ',' '\n' | wc -l); mkdir -p $O $F; H=$S/lingbot-world-v2/specwm_out/drafthead/rt2c1_bl
# [10/17 C] H = rt2c1_bl: the rt2c1 step-1 heads + thresholds (symlinks, identical) with result.json 'folds' extended by every baselines/tasks/*.json id (scene in a
#   StEvo fold -> that held-out fold; wp/mg2/abot/astra scenes -> fold 0). With plain rt2c1 two_machine's exact-id fold lookup raises StopIteration on every
#   baseline task (found in the adapt25c1hyb smoke 10/17 08:31; affects the solo adapt25 arm too, which had not run yet).
W=18; [ $A -gt 600000 ] && W=12; T5OPT=""; [ $A -gt 600000 ] && T5OPT="--t5-cpu"   # spec_early arms: text encoder on the CPU at 1280x704 (11 GiB)      # two_machine KV window (spec_early arms: fixed 18, see WREC)
export SPECWM_TASKS=$J CUDA_VISIBLE_DEVICES=$G SAVE_LAT=0
WQ=fp8; WREC=$W; KIND=two_machine; XD=; WD=   # [10/19] XD = extra DEVICE-only flags (appended last: a later --window wins), WD = device KV window for the dev conversion (rt2c1hyb32)
WQD=   # [10/19 thesis] device weight format for the dev-stream conversion (empty = fp8, unchanged); gptq4 for rt2c1hyb24 / mxQ4* / mx24*
GQ=$S/specwm_out/gptq4/14b_g128_832.pt; LDEC="--lowmem 1 --vae-dtype bf16 --vae-tiles 2,2"   # [10/19 thesis] the offline GPTQ-4 file (specwm.gptq_save, gen/queue/8/310) + the light-device decode flags
mx_flags() {   # [10/19 thesis] MATRIX table: name -> "device flags|window for the record|device wq" (one squeeze at a time on top of rt2, then the two bundles)
  case $1 in
    mxW12) echo "--window 12|12|fp8";; mxW9) echo "--window 9|9|fp8";; mxK8) echo "--kv fp8|18|fp8";; mxQ4) echo "--wq gptq4 --wq-file $GQ --lowmem 1|18|gptq4";;
    mx32) echo "--window 12 --kv fp8 --cap 32 $LDEC|12|fp8";; mx24) echo "--window 12 --kv fp8 --cap 24 $LDEC --wq gptq4 --wq-file $GQ|12|gptq4";; *) return 1;;
  esac; }
case $ARM in
  teacher) X="--nodes 0,250,500,750 --use 0,1,2,3";; rt2) X="--nodes 0,500 --use 0,1";; solo3) X="--nodes 0,500,750 --use 0,1,2";;
  adapt25) X="--nodes 0,500,750 --use 0,1 --adapt-budget 0.25 --adapt-node 2 --gate-dir $H";;
  adapt25c1hyb) KIND=hyb; X="--nodes 0,500,750 --use 0,1 --adapt-budget 0.25 --adapt-node 2 --gate-dir $H --thr 0"; XC="--nodes 0,250,500,750 --use 3 --max-lag 99";;
  rt2c1hyb) KIND=hyb; X="--nodes 0,500 --use 0,1 --thr 0"; XC="--nodes 0,250,500,750 --use 3 --max-lag 99";;   # [10/19 EXP 2] = specwm_out/twomach rt2c1hyb (device --nodes 0,500 --use 0,1 --thr 0; cloud default nodes, --use 3 --max-lag 99)
  rt2c1hyb32) KIND=hyb; X="--nodes 0,500 --use 0,1 --thr 0"; XC="--nodes 0,250,500,750 --use 3 --max-lag 99"; XD="--window 12 --kv fp8 --cap 32 --lowmem 1 --vae-dtype bf16 --vae-tiles 2,2"; WD=12;;   # [10/19 EXP 2 light device] rt2c1hyb + 32-GiB device
  rt2c1hyb24) KIND=hyb; X="--nodes 0,500 --use 0,1 --thr 0"; XC="--nodes 0,250,500,750 --use 3 --max-lag 99"; XD="--window 12 --kv fp8 --cap 24 --lowmem 1 --vae-dtype bf16 --vae-tiles 2,2 --wq gptq4 --wq-file $GQ"; WD=12; WQD=gptq4;;   # [10/19 thesis] rt2c1hyb32 + GPTQ-4 weights, 24-GiB device
  rt2c1hyb32o50|rt2c1hyb32o25|rt2c1hyb32o12)   # [10/22 step 0, additive] ORACLE gate: rt2c1hyb32 (every chunk sent, cloud refines every chunk) but the device re-commits a correction only if the cloud CHANGED the chunk by more than --thr (|x_corr - x_draft|/|x_draft| > thr, thr = the 50 / 75 / 87.5 percentile of rel in the frozen rt2c1hyb32 runs of the same set, so ~50 / 25 / 12.5 % of the corrections are applied). Upper bound on "quality at a given correction fraction" (cloud cost NOT saved); the random gate g<B> is the lower bound.
    case "$SET:$ARM" in mg2*:*o50) OT=0.252;; mg2*:*o25) OT=0.283;; mg2*:*o12) OT=0.300;; *:*o50) OT=0.268;; *:*o25) OT=0.291;; *:*o12) OT=0.305;; esac
    KIND=hyb; X="--nodes 0,500 --use 0,1 --thr $OT"; XC="--nodes 0,250,500,750 --use 3 --max-lag 99"; XD="--window 12 --kv fp8 --cap 32 --lowmem 1 --vae-dtype bf16 --vae-tiles 2,2"; WD=12;;
  rt2c1hyb32d)   # [10/22 step 0, additive] LEARNED gate: rt2c1hyb32 whose device sends a chunk only when the 10/9 scene-general drift head (layer-20 hidden of the dense commit pass, specwm_out/drift_head/head_L20_all.pt, --drift-tau -3.25 = the 10/10 calibration) flags it (chunks 0-2 always); the rest accepted as drafted. The realized send fraction is in timing.json (chunks_sent).
    KIND=hyb; X="--nodes 0,500 --use 0,1 --thr 0 --gate drift --drift-tau -3.25"; XC="--nodes 0,250,500,750 --use 3 --max-lag 99"; XD="--window 12 --kv fp8 --cap 32 --lowmem 1 --vae-dtype bf16 --vae-tiles 2,2"; WD=12;;
  rt2c1hyb32L1)   # [10/22 step 3, additive] rt2c1hyb32 with a FINITE cloud backlog: --max-lag 1 on the cloud (two_machine default): when more than one draft is queued the cloud commits the draft as is (mode skip) instead of refining stale chunks, so a slow cloud degrades gracefully instead of letting every correction age out (10/22 stale-cloud incident: --max-lag 99, lag 10-17 s, 0 applied).
    KIND=hyb; X="--nodes 0,500 --use 0,1 --thr 0"; XC="--nodes 0,250,500,750 --use 3 --max-lag 1"; XD="--window 12 --kv fp8 --cap 32 --lowmem 1 --vae-dtype bf16 --vae-tiles 2,2"; WD=12;;
  rt2c1hyb32g50|rt2c1hyb32g25|rt2c1hyb32g12|rt2c1hyb32g0)   # [10/21 cloud-fraction sweep, additive] rt2c1hyb32 + two_machine --gate random --gate-budget B on the DEVICE: chunks 0-2 always sent, later chunks sent with probability B (seeded RNG, --seed), the rest accepted as drafted (the cloud commits them as is, no correction returned). g50 B=0.5, g25 0.25, g12 0.125, g0 0 (chunks 0-2 only)
    GB=${ARM#rt2c1hyb32g}; case $GB in 50) GB=0.5;; 25) GB=0.25;; 12) GB=0.125;; 0) GB=0.0;; esac
    KIND=hyb; X="--nodes 0,500 --use 0,1 --thr 0 --gate random --gate-budget $GB"; XC="--nodes 0,250,500,750 --use 3 --max-lag 99"; XD="--window 12 --kv fp8 --cap 32 --lowmem 1 --vae-dtype bf16 --vae-tiles 2,2"; WD=12;;
  mx*)   # [10/19 thesis] MATRIX: mx<X> (solo) / mx<X>c1hyb (pair); one branch, the table above
    MXB=${ARM%c1hyb}; MXF=$(mx_flags $MXB) || { echo "unknown arm $ARM (matrix names: mxW12 mxW9 mxK8 mxQ4 mx32 mx24, + c1hyb)"; exit 1; }; IFS='|' read -r XD WD WQD <<< "$MXF"
    X="--nodes 0,500 --use 0,1"
    if [ "$MXB" != "$ARM" ]; then KIND=hyb; X="$X --thr 0"; XC="--nodes 0,250,500,750 --use 3 --max-lag 99"
    else WQ=$WQD; WREC=$WD; case " $XD " in *" --cap "*) TIMED_DECODE=0;; esac; fi;;   # solo: the conversion records the device window / wq; capped solo arms: no side-stream decode
  sf24c1hyb) KIND=hyb; X="--nodes 0,250,500,750 --use 0,1,2,3 --thr 0 --commit separate --ee-head $S/lingbot-world-v2/specwm_out/head14b_nob43/head.pt --ee-eps 0.24 --ee-bucket 128"; XC="--nodes 0,250,500,750 --use 3 --max-lag 99";;   # [10/19 EXP 3] device = the sf24 arm inside two_machine
  sf24) KIND=spec_early; WREC=$W;;
  student1p3b) KIND=spec_early; WQ=bf16; WREC=$W;;
  sf1p3b_pub|sf1p3b_cal)
    KIND=spec_early_tf; WQ=bf16; WREC=18
    [ "$A" -eq 399361 ] || { echo "arm $ARM runs only at 832x480 (area 399361: sets astra, neutral_832, neutral_832_nocap); $SET has area $A"; exit 1; }
    KEY=published; [ $ARM = sf1p3b_cal ] && KEY=calibrated
    read TAU_RED TAU_STAB <<< "$($R python -c "import json; c = json.load(open('$B/calib.json'))['sf1p3b']['$KEY']; print(c['tau_red'], c['tau_stab'])" 2>/dev/null | tail -1)"
    [ -n "$TAU_STAB" ] || { echo "could not read calib.json['sf1p3b']['$KEY']"; exit 1; };;
  *) echo "unknown arm $ARM"; exit 1;;
esac
LOG=$B/logs/ours_${ARM}_${SET}${RUNS_SUFFIX}.log   # [10/19 audit] suffixed like the dirs
echo "[$(date +%F_%H:%M)] ours $ARM $SET${RUNS_SUFFIX:+ suffix $RUNS_SUFFIX} area $A frames $FRAMES window $WREC gpu $G tasks $J ($NT to run, $NDONE already frozen)${TAU_RED:+ tau_red $TAU_RED tau_stab $TAU_STAB}${RUNS_SUFFIX:+ stream_decode $STREAM_DECODE}${XD:+ device-only flags: $XD (wq ${WQD:-fp8}, timed_decode $TIMED_DECODE)}" | tee -a $B/logs/run_ours.out | tee -a $LOG
if [ $ARM = sf24 ]; then $R python -m specwm.spec_early --tasks $T --seeds 1234 --out $O --arm sf24 --area $A --frames $FRAMES --window $W $T5OPT --head $S/lingbot-world-v2/specwm_out/head14b_nob43/head.pt --bucket 128 --warm 1 --mode early --eps 0.24 --stream-decode $STREAM_DECODE $LONGSE >> $LOG 2>&1
elif [ $ARM = student1p3b ]; then $R python -m specwm.spec_early --tasks $T --seeds 1234 --out $O --arm student1p3b --area $A --frames $FRAMES --window $W $T5OPT --model 1p3b --warm 1 --mode uniform:4 --stream-decode $STREAM_DECODE $LONGSE >> $LOG 2>&1
elif [ $KIND = hyb ]; then   # [10/17 C] cloud on GPU LO first, device on GPU HI once the cloud listens; both get the same SPECWM_TASKS / --tasks / --area / --frames / --window
  PORT=$((32300 + LO)); LOGC=$B/logs/ours_${ARM}_${SET}${RUNS_SUFFIX}_cloud.log; LOGD=$B/logs/ours_${ARM}_${SET}${RUNS_SUFFIX}_device.log; mkdir -p $O/cloud $O/device
  HX="--wq fp8 --tasks $T --area $A --frames $FRAMES --window $W $LONG"; CP=; DP=; PP=; RC=0; PKF=$(mktemp); th0=$(date +%s)
  hyb_cleanup() { for p in $DP $CP $PP; do kill -0 $p 2>/dev/null && kill $p 2>/dev/null; done; for p in $DP $CP $PP; do wait $p 2>/dev/null; done; }
  trap 'hyb_cleanup' EXIT; trap 'exit 143' TERM INT HUP
  echo "[$(date +%F_%T)] hybrid pair: cloud gpu $LO port $PORT -> $LOGC ; device gpu $HI -> $LOGD${XD:+ ; device-only flags: $XD}" | tee -a $LOG
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i $LO,$HI -l 2 > $PKF 2>/dev/null & PP=$!
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$PORT\$"; then echo "port $PORT already in use -> not starting" | tee -a $LOG; RC=4
  else
    n0=$(cat $LOGC 2>/dev/null | wc -l)
    CUDA_VISIBLE_DEVICES=$LO $R python -m specwm.two_machine --role cloud --port $PORT --out $O/cloud $XC $HX >> $LOGC 2>&1 & CP=$!
    for i in $(seq 1 450); do   # <= 15 min for the cloud's model load until it listens (bind + listen happen before its warm-up)
      tail -n +$((n0 + 1)) $LOGC 2>/dev/null | grep -q "listening on $PORT" && break
      kill -0 $CP 2>/dev/null || break; sleep 2; done
    if ! kill -0 $CP 2>/dev/null; then wait $CP; RC=$?; echo "cloud died before listening (rc $RC)" | tee -a $LOG; [ $RC -eq 0 ] && RC=5
    elif ! tail -n +$((n0 + 1)) $LOGC | grep -q "listening on $PORT"; then echo "cloud not listening after 900 s -> abort" | tee -a $LOG; RC=5
    else
      CUDA_VISIBLE_DEVICES=$HI $R python -m specwm.two_machine --role device --connect 127.0.0.1:$PORT --out $O/device $X $HX $XD >> $LOGD 2>&1 & DP=$!
      wait $DP; RC=$?; DP=
      if [ $RC -ne 0 ]; then echo "device failed rc $RC -> killing the cloud (pid $CP)" | tee -a $LOG; kill $CP 2>/dev/null; wait $CP 2>/dev/null; CP=
      else
        for i in $(seq 1 150); do kill -0 $CP 2>/dev/null || break; sleep 2; done   # the cloud exits on the device's END (after its last decode)
        if kill -0 $CP 2>/dev/null; then echo "cloud still running 300 s after the device finished -> killed" | tee -a $LOG; kill $CP; wait $CP 2>/dev/null; RC=6
        else wait $CP; CRC0=$?; [ $CRC0 -ne 0 ] && { echo "cloud rc $CRC0" | tee -a $LOG; RC=$CRC0; }; fi; CP=
      fi
    fi
  fi
  kill $PP 2>/dev/null; wait $PP 2>/dev/null; PP=
  PK=$(awk -F', *' '{ if ($2 + 0 > m[$1]) m[$1] = $2 + 0 } END { for (k in m) printf "gpu%s %d MiB ", k, m[k] }' $PKF); rm -f $PKF
  echo "HYB_HOLD $SET cloud gpu $LO device gpu $HI: hold $(( $(date +%s) - th0 )) s for $NT clips, nvidia-smi peak $PK, rc $RC" | tee -a $LOG | tee -a $B/logs/run_ours.out
elif [ $KIND = spec_early_tf ]; then $R python -m specwm.spec_early_tf --tasks $T --seeds 1234 --out $O --arm $ARM --tasks-json $J --area $A --frames $FRAMES --model 1p3b --mode tf --tau-red $TAU_RED --tau-stab $TAU_STAB --bucket 128 --warm 1 --stream-decode $STREAM_DECODE >> $LOG 2>&1
else $R python -m specwm.two_machine --role solo --out $O --wq fp8 $X --tasks $T --area $A --frames $FRAMES --window $W --timed-decode $TIMED_DECODE $LONG $XD >> $LOG 2>&1; fi   # [10/19 thesis] $XD appended last (empty for every older arm)
RC0=$?; [ $KIND = hyb ] || RC=$RC0
if [ $KIND = hyb ]; then   # [10/17 C] two frozen arms; each conversion reads the partner stream's raw timing.json (kept by --purge)
  $R python $B/ours_to_timing.py $O/device $F --batch --arm ${ARM}_dev --set $SET --kv-window ${WD:-$W} --wq ${WQD:-fp8} --tasks $T --purge 2>&1 | tee -a $LOG; CRC=${PIPESTATUS[0]}   # [10/19 thesis] WQD: gptq4 device arms
  $R python $B/ours_to_timing.py $O/cloud $FC --batch --arm ${ARM}_cloud --set $SET --kv-window $W --wq fp8 --tasks $T --purge 2>&1 | tee -a $LOG; CRC2=${PIPESTATUS[0]}
  [ $CRC2 -ne 0 ] && CRC=$CRC2
else
$R python $B/ours_to_timing.py $O $F --batch --arm $ARM --set $SET --kv-window $WREC --wq $WQ --tasks $T --purge 2>&1 | tee -a $LOG; CRC=${PIPESTATUS[0]}
fi
echo "OURS_DONE $ARM $SET rc $RC conv_rc $CRC $(date +%F_%H:%M) -> $F" | tee -a $B/logs/run_ours.out | tee -a $LOG
[ $RC -ne 0 ] && exit $RC; [ $CRC -ne 0 ] && exit 2; exit 0
