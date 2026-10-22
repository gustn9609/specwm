#!/bin/bash
# [10/19 thesis] tier 8: OFFLINE GPTQ-4 checkpoint of the 14B causal-fast DiT blocks -> /home/hk28965/specwm/specwm_out/gptq4/14b_g128_832.pt (~7.5 GB)
#   specwm/gptq_save.py: bf16 14B on the GPU (34.5 GiB, fit24's load, no compile / no FP8), Hessians (X^T X per block linear, host RAM) from 3 chunks of EACH of
#   2 calibration scenes that are NOT benchmark scenes (StEvo ball_roll_down + bouncing_ball_rebound of specwm_out/refiner/train_tasks.txt) at 832x480 (area 399361)
#   on the rt2 schedule (nodes 0,500, clean commit pass included), KV window 18; then GPTQ 4-bit g128 per linear (non-finite at damp 0.01 -> redone at 0.1, as fit24),
#   packed to the int4_kernels storage, saved with metadata; reloaded and checked (0 non-finite layers). Idempotent: an existing file that loads clean is kept.
#   Refuses when /home has < 20 GB free (exit 3; the user approves deletions). fit24's in-process GPTQ at 832 took ~13 min; expect ~20-30 min incl. the load.
# usage: bash THIS GPU   (worker.sh runs .sh chunks WITHOUT a lock; this script takes flock /tmp/gpu$G.lock itself, exclusive, one hold; GPUs 0-3 only).
# sentinel: gen/sentinels/gptq4_save.ok (file exists + loads + non-finite layers 0; JSON with the meta) | .fail (why). The tier-8 memory check (311) and
#   the tier-9 rt2c1hyb24 / mx24* / mxQ4* chunks gate on it (gen/make_queue_matrix.py).
G=${1:?usage: $0 GPU}; S=/home/hk28965/specwm; B=$S/baselines; NAME=310_gptq4_save; L=$B/gen/logs/$NAME.log; SENT=$B/gen/sentinels; OUT=$S/specwm_out/gptq4/14b_g128_832.pt; mkdir -p $B/gen/logs $SENT
case "$G" in 0|1|2|3|4|5|6) ;; *) echo "[queue] $NAME: GPU $G not allowed"; exit 1;; esac   # [10/20 23:58 copy in paper/1020/chains: GPUs 0-6 re-allowed after the reboot]
t0=$(date +%s); PK=$(mktemp); echo "[queue $(date '+%m/%d %H:%M:%S')] $NAME: GPTQ-4 save -> $OUT, waiting for /tmp/gpu$G.lock" | tee -a $L
flock /tmp/gpu$G.lock env CUDA_VISIBLE_DEVICES=$G $B/_gpu_peak_run.sh $G "$PK" $S/scripts/envrun.sh python -m specwm.gptq_save --out $OUT >> $L 2>&1; RC=$?
read -r HELD PEAK < "$PK" 2>/dev/null; rm -f "$PK"
echo "[queue $(date '+%m/%d %H:%M:%S')] $NAME: rc $RC, wall $(( $(date +%s) - t0 )) s, lock hold ${HELD:-?} s, peak GPU memory ${PEAK:-?} MiB" | tee -a $L
# verify on the CPU: exists + loads (weights_only) + 0 non-finite layers -> sentinel
CUDA_VISIBLE_DEVICES= $S/scripts/envrun.sh python -I - "$OUT" "$SENT/gptq4_save" "$RC" "${HELD:-0}" "${PEAK:-0}" "$NAME" <<'EOF_CHECK' 2>&1 | tee -a $L; exit ${PIPESTATUS[0]}
import json, os, sys, time
sys.path.insert(0, "/home/hk28965/specwm/lingbot-world-v2")
out, sent, rc, held, peak, name = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
d = dict(name=name, out=out, rc=rc, hold_s=held, gpu_peak_MiB=peak, checked=time.strftime("%F %T")); bad = []
if rc != 0: bad.append("gptq_save rc %d" % rc)
if not os.path.exists(out): bad.append("no file %s" % out)
else:
    d["size_GB"] = round(os.path.getsize(out) / 1e9, 3)
    try:
        from specwm.int4_packed import check_file
        meta, nl, nf = check_file(out); d.update(n_layers=nl, nonfinite_layers=nf, meta={k: meta.get(k) for k in ("calib_tasks", "resolution", "nodes", "window", "seed", "damp", "redamped", "nonfinite", "quant_s", "calib_s", "total_s", "peak_GiB", "made", "scope")})
        if nf: bad.append("%d non-finite layers: %s" % (len(nf), nf[:5]))
        if nl < 500: bad.append("only %d layers (expected ~568 block linears)" % nl)
    except Exception as e: bad.append("does not load: %r" % e)
d["ok"] = not bad
if bad:
    d["problems"] = bad; json.dump(d, open(sent + ".fail", "w"), indent=1); os.path.exists(sent + ".ok") and os.remove(sent + ".ok"); print("[gptq4_save] FAIL", bad); sys.exit(1)
json.dump(d, open(sent + ".ok.tmp", "w"), indent=1); os.replace(sent + ".ok.tmp", sent + ".ok"); os.path.exists(sent + ".fail") and os.remove(sent + ".fail")
print("[gptq4_save] ok: %d layers, %.2f GB, redamped %s" % (d["n_layers"], d["size_GB"], d["meta"].get("redamped"))); sys.exit(0)
EOF_CHECK
