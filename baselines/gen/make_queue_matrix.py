#!/usr/bin/env python3
"""[10/19 thesis] Queue tier 9 (fresh dir gen/queue/9; the old tier 9 is parked): the memory-squeeze MATRIX on the 60-s P6 sets.
Thesis: a memory-squeezed on-device world model (small KV window, FP8 KV, 4-bit weights, 2 steps) drifts over 60-s rollouts; a full-context cloud that
re-anchors the device KV repairs it. Sets neutral_832_long (832x480) and mg2_long (640x352), 4 scenes each (tasks/<set>_long.json, 969 poses -> 961 frames).
Arms (run_ours.sh mx_flags; every device runs the rt2 schedule nodes 0,500 --use 0,1): device-only mxW12 mxW9 mxK8 mxQ4 mx32 mx24 (two_machine --role solo)
and the hybrids mx<X>c1hyb (same device flags + the standard cloud: nodes 0,250,500,750 --use 3 --max-lag 99, --thr 0) + the 60-s versions of the light
devices rt2c1hyb32 / rt2c1hyb24. NOTE mx32c1hyb == rt2c1hyb32 and mx24c1hyb == rt2c1hyb24 flag for flag (same seed -> the same computation), so the queue
runs rt2c1hyb32 / rt2c1hyb24 and does NOT duplicate them as mx32c1hyb / mx24c1hyb (both arm names exist in run_ours.sh; --dup-hyb queues them anyway).
Order: 340/341 = the rt2c1hyb24 pair SMOKES (10-s, one clip, mg2 then neutral_832; smoke_check.py --kind hyb + the light check with cap 24 and the gptq4
fields -> gen/sentinels/rt2c1hyb24_<set>.ok|.fail), gated on the tier-8 memory check mem24_<res>.ok (311, itself gated on gptq4_save.ok = 310); then from
342 the 60-s matrix, mg2_long first (cheaper, its smoke is first), every chunk gated on the smoke of its resolution class (rt2c1hyb24_mg2 for 640x352,
rt2c1hyb24_neutral_832 for 832x480) and the gptq4 arms (mx24*, mxQ4*, rt2c1hyb24) on gptq4_save.ok too.
Gate semantics (gate NAME GLOB): .ok -> go; .fail -> exit 1 at once; no sentinel and the gating chunk (GLOB) neither pending in any tier nor claimed -> exit 1;
else wait without a lock up to GATE_WAIT (default 2000 s) and then DEFER = copy itself back to gen/queue/9/ and exit 0 (the worker's CHUNK_TIMEOUT 5400 s
counts the wait, so a chunk never fails only because its smoke is slow; 2000 + the largest chunk estimate 2920 < 5400).
Chunks: solo arms = one chunk per (arm, set) = 4 long clips, written as the tier-6 single-GPU wrapper (.sh with the embedded chunk json: gate first, then
flock gpu$G + _gpu_peak_run.sh + run_ours.sh SET ARM G CHUNKJSON; a bare .json chunk cannot gate); hybrids = the tier-6 pair .sh chunks (flock gpu<lo> then
gpu<hi>, cloud on lo / device on hi, GPUs 0-3 only), split so that load + n x clip <= TARGET 3000 s. Per-clip estimates (s) are derived from the measured
tier-5/6 long clips (teacher 832x480 P6: 233.5 s wall for prepare 35 + gen 181 + decode 53; load + 81-position warm-up ~660 s) and the frozen 10-s runs
(rt2 / teacher gen ratio 0.55 at 832, 0.55 at 640; rt2c1hyb pair hold 61 s/clip at 832 and 32 s/clip at 640 vs rt2 solo e2e 32 / 17 s = x1.9), see EST.
usage: python3 /home/hk28965/specwm/baselines/gen/make_queue_matrix.py [--dry-run] [--dup-hyb] [--start 340]   (plain python3; re-running adds only
clips not yet queued / frozen; writes gen/queue/plan_matrix_1019.json and appends INDEX.tsv)"""
import os, sys, json, math, time, glob, argparse
B = "/home/hk28965/specwm/baselines"; GEN = B + "/gen"; S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--dup-hyb", action="store_true", help="also queue mx32c1hyb / mx24c1hyb (identical computations to rt2c1hyb32 / rt2c1hyb24)")
ap.add_argument("--start", type=int, default=340); ap.add_argument("--queue-root", default=GEN + "/queue"); ap.add_argument("--nvsmi-max", type=int, default=24564, help="smoke light check: max device-process / device-GPU nvidia-smi MiB (24564 = an RTX 4090's total)")
a = ap.parse_args(); Q = a.queue_root; TIER = 9; MADE = time.strftime("%Y-%m-%d %H:%M:%S")
SETS = ["mg2_long", "neutral_832_long"]; SMOKE_SET = {"mg2_long": "mg2", "neutral_832_long": "neutral_832"}; RESI = {"832x480": 0, "640x352": 1}
GQ = S + "/specwm_out/gptq4/14b_g128_832.pt"
# ---- per-clip estimates (s): (832x480, 640x352) one 60-s clip inside a warm hold incl. its conversion; LOAD = load + warm-up on all 81 chunk positions ----
EST = {"mxW12": (200, 115), "mxW9": (195, 110), "mxK8": (215, 120), "mxQ4": (265, 150), "mx32": (230, 130), "mx24": (290, 165),
       "mxW12c1hyb": (380, 220), "mxW9c1hyb": (370, 210), "mxK8c1hyb": (410, 230), "mxQ4c1hyb": (500, 285), "mx32c1hyb": (440, 250), "mx24c1hyb": (550, 315), "rt2c1hyb32": (440, 250), "rt2c1hyb24": (550, 315)}
LOAD = {k: (720 if not k.endswith("c1hyb") and not k.startswith("rt2c1hyb") else 1400) + (80 if ("Q4" in k or "24" in k or "32" in k) else 0) for k in EST}   # lowmem / gptq4 loads: + CPU load + block-wise move
EST10 = {"rt2c1hyb24": (75, 40)}; LOAD10 = {"rt2c1hyb24": 1500}   # the 10-s smoke: rt2c1hyb's tier-6 51 / 27 s per clip + the GPTQ-4 dequant and the frame-by-frame bf16 decode
TARGET, CAP = 3000, 4
GATE_SH = r'''SENT=$B/gen/sentinels; QR=$B/gen/queue; GATE_WAIT=${GATE_WAIT:-2000}; mkdir -p $SENT
gate() {   # $1 = sentinel name, $2 = its chunk's file glob: go on when $SENT/$1.ok exists; exit 1 at once on $SENT/$1.fail or when the chunk is neither pending (any tier) nor running; else wait (no lock held)
  local n=$1 g=$2 t0=$(date +%s)
  while :; do
    [ -f $SENT/$n.ok ] && { echo "[gate] $n ok"; return 0; }
    [ -f $SENT/$n.fail ] && { echo "[gate] $n FAILED: $(head -c 400 $SENT/$n.fail | tr '\n' ' ') -> exit 1"; exit 1; }
    [ -n "$(ls $QR/[0-9]*/$g $QR/claimed/$g.gpu* 2>/dev/null)" ] || { echo "[gate] $n: no sentinel and its chunk ($g) is neither pending nor running (failed / killed?) -> exit 1"; exit 1; }   # [10/19 thesis] -n "$(ls ...)": plain `ls A B` exits 2 when ONE glob is unmatched
    [ $(( $(date +%s) - t0 )) -ge $GATE_WAIT ] && { echo "[gate] $n still pending/running after ${GATE_WAIT} s -> DEFER: this chunk re-queues itself as $QR/9/$NAME.sh and exits 0 (the worker's CHUNK_TIMEOUT 5400 s counts the wait)"; cp -- "$0" $QR/9/$NAME.sh.tmp && chmod +x $QR/9/$NAME.sh.tmp && mv -f $QR/9/$NAME.sh.tmp $QR/9/$NAME.sh; exit 0; }
    sleep 30
  done
}'''
GATES = {"gptq4_save": "*_gptq4_save.sh", "mem24_640x352": "*_light_mem_check.sh", "mem24_832x480": "*_light_mem_check.sh", "rt2c1hyb24_mg2": "*_smoke_rt2c1hyb24_mg2.sh", "rt2c1hyb24_neutral_832": "*_smoke_rt2c1hyb24_neutral_832.sh"}
LIGHT_CHECK = r'''
if [ $RC -eq 0 ]; then   # [10/19 thesis] light-device check on top of smoke_check.py: the device really ran capped at 24 GiB with GPTQ-4 weights / FP8 KV / window 12 and stayed below NVSMI_MAX
  python3 -I - "$B/runs/ours_rt2c1hyb24_dev_@SET@" "@IDS@" "$SENT/@SMOKE@" @NVMAX@ "$B/logs/ours_rt2c1hyb24_@SET@.log" <<'EOF_LIGHT'; RC=$?
import json, os, re, sys
runs, ids, sent, nvmax, hlog = sys.argv[1], sys.argv[2].split(","), sys.argv[3], int(sys.argv[4]), sys.argv[5]; bad = []; got = {}
hold = [ln for ln in open(hlog) if ln.startswith("HYB_HOLD ")] if os.path.exists(hlog) else []   # run_ours.sh: GPU-wide nvidia-smi peaks of the pair over the hold (2-s polling)
m = re.search(r"device gpu (\d+):.*nvidia-smi peak (.*), rc", hold[-1]) if hold else None
dpk = int(re.search(r"gpu%s (\d+) MiB" % m.group(1), m.group(2)).group(1)) if m and re.search(r"gpu%s (\d+) MiB" % m.group(1), m.group(2)) else None
got["hyb_hold_device_gpu_peak_MiB"] = dpk
if dpk is None or dpk > nvmax: bad.append("HYB_HOLD device-GPU nvidia-smi peak %s MiB (whole GPU, whole hold) > %d or missing" % (dpk, nvmax))
for i in ids:
    T = json.load(open(os.path.join(runs, i, "timing.json"))); M = T.get("device_mem") or {}
    got[i] = dict(cap=M.get("cap_GiB"), kv=M.get("kv"), window=M.get("window"), lowmem=M.get("lowmem"), wq=M.get("wq"), int4_kernel=M.get("int4_kernel"), wq_layers=M.get("wq_layers"), vae_dtype=M.get("vae_dtype"), vae_tiles=M.get("vae_tiles"),
                  proc_peak_alloc_GiB=M.get("proc_peak_alloc_GiB"), proc_peak_reserved_GiB=M.get("proc_peak_reserved_GiB"), nvsmi_peak_MiB=M.get("nvsmi_proc_peak_MiB"), kv_window=T.get("kv_window"), wq_rec=T.get("wq"), precision=(T.get("precision") or "")[:40], fps_shown=T.get("device_fps_shown"))
    if not M: bad.append("%s: device timing has no mem record" % i); continue
    if M.get("cap_GiB") != 24.0 or M.get("kv") != "fp8" or M.get("window") != 12 or M.get("lowmem") != 1 or M.get("wq") != "gptq4" or not M.get("int4_kernel") or M.get("vae_dtype") != "bf16" or M.get("vae_tiles") != "2,2": bad.append("%s: device flags %s" % (i, got[i]))
    if T.get("wq") != "gptq4" or T.get("kv_window") != 12: bad.append("%s: dev conversion records wq %s / kv_window %s (expected gptq4 / 12)" % (i, T.get("wq"), T.get("kv_window")))
    if (M.get("proc_peak_reserved_GiB") or 99) > 24.0: bad.append("%s: device reserved peak %s GiB > 24" % (i, M.get("proc_peak_reserved_GiB")))
    if M.get("nvsmi_proc_peak_MiB") is None or M["nvsmi_proc_peak_MiB"] > nvmax: bad.append("%s: device nvidia-smi peak %s MiB > %d" % (i, M.get("nvsmi_proc_peak_MiB"), nvmax))
ok = sent + ".ok"; fail = sent + ".fail"
if bad:
    d = json.load(open(ok)) if os.path.exists(ok) else {}; d.update(ok=False, light_problems=bad, light=got)
    json.dump(d, open(fail, "w"), indent=1); os.path.exists(ok) and os.remove(ok); print("[light] FAIL", bad); sys.exit(1)
d = json.load(open(ok)); d["light"] = got; json.dump(d, open(ok + ".tmp", "w"), indent=1); os.replace(ok + ".tmp", ok); print("[light] ok", got); sys.exit(0)
EOF_LIGHT
fi'''
def gate_calls(gates): return "\n".join("gate %s \"%s\"" % (g, GATES[g]) for g in gates)
def ours_script(name, d, gates):
    """single-GPU OUR arm chunk (= make_queue_long.ours_script, tier 9, generalized gate)"""
    stem = name[:-3]; js = json.dumps(d, indent=1); assert "\nEOF_CHUNK\n" not in js
    return """#!/bin/bash
# generated by gen/make_queue_matrix.py %(made)s -- tier %(tier)d: OUR %(arm)s (matrix, device only) on %(set)s, %(n)d clips of 60 s at %(res)s, ONE GPU (re-run make_queue_matrix.py, do not edit)
# OURS_CHUNK set=%(set)s arm=%(arm)s n=%(n)d gates=%(gates)s ids=%(ids)s
# usage: bash THIS GPU   (worker.sh runs .sh chunks so, WITHOUT a lock): gate on the sentinels first (no lock held while waiting), then
# flock /tmp/gpu$G.lock env CUDA_VISIBLE_DEVICES=$G _gpu_peak_run.sh run_ours.sh %(set)s %(arm)s $G CHUNKJSON (skip-if-done; FRAMES = the tasks' 969 -> --long-clip 1).
G=${1:?usage: $0 GPU}; B=/home/hk28965/specwm/baselines; NAME=%(stem)s
case "$G" in 0|1|2|3) ;; *) echo "[queue] $NAME: GPU $G: tier 9 runs only on GPUs 0-3 (10/19: GPUs 4-7 not ours)"; exit 1;; esac
%(gate_sh)s
%(gate_calls)s
CJ=$B/raw/ours_%(arm)s_%(set)s/chunkdef_$NAME.json; mkdir -p $B/raw/ours_%(arm)s_%(set)s
cat > $CJ.tmp <<'EOF_CHUNK'
%(js)s
EOF_CHUNK
mv -f $CJ.tmp $CJ
t0=$(date +%%s); PK=$(mktemp); echo "[queue $(date '+%%m/%%d %%H:%%M:%%S')] $NAME: %(n)d clips, waiting for /tmp/gpu$G.lock"
flock /tmp/gpu$G.lock env CUDA_VISIBLE_DEVICES=$G $B/_gpu_peak_run.sh $G "$PK" $B/run_ours.sh %(set)s %(arm)s $G $CJ; RC=$?
read -r HELD PEAK < "$PK" 2>/dev/null; rm -f "$PK"
echo "[queue $(date '+%%m/%%d %%H:%%M:%%S')] $NAME: rc $RC, wall $(( $(date +%%s) - t0 )) s, lock hold ${HELD:-?} s, peak GPU memory ${PEAK:-?} MiB"
exit $RC
""" % dict(made=MADE, tier=TIER, arm=d["arm"], set=d["set"], n=d["n"], res=d["res"], gates=",".join(gates) or "-", ids=",".join(d["ids"]), stem=stem, gate_sh=GATE_SH if gates else "", gate_calls=gate_calls(gates), js=js)
def hyb_script(name, d, gates, smoke=None):
    """GPU-pair hybrid chunk (= make_queue_long.hyb_script + generalized gate + the rt2c1hyb24 smoke checks)"""
    stem = name[:-3]; js = json.dumps(d, indent=1); assert "\nEOF_CHUNK\n" not in js; arm = d["arm"]
    chk = ""
    if smoke:
        chk = ("\npython3 $B/gen/smoke_check.py %s --kind hyb --runs $B/runs/ours_%s_dev_%s --runs-cloud $B/runs/ours_%s_cloud_%s --ids %s --frames 161 --rc $RC --note \"$NAME (see HYB_HOLD in logs/ours_%s_%s.log)\"; RC=$?"
               % (smoke, arm, d["set"], arm, d["set"], ",".join(d["ids"]), arm, d["set"])) + LIGHT_CHECK.replace("@SET@", d["set"]).replace("@IDS@", ",".join(d["ids"])).replace("@NVMAX@", str(a.nvsmi_max)).replace("@SMOKE@", smoke)
    return """#!/bin/bash
# generated by gen/make_queue_matrix.py %(made)s -- tier %(tier)d: OUR hybrid %(arm)s on %(set)s, %(n)d clips%(dur)s at %(res)s, on a GPU PAIR%(smk)s (re-run make_queue_matrix.py, do not edit)
# HYB_CHUNK set=%(set)s arm=%(arm)s n=%(n)d gates=%(gates)s ids=%(ids)s
# usage: bash THIS GPU   (worker.sh runs .sh chunks so, WITHOUT a lock): gate on the sentinels first (no lock held while waiting), then the tier-4/6
# pair mechanics: pair of G = G xor 1 (ONLY 0+1 and 2+3: GPUs 4-7 are not ours; G 4-7 exits 1), cloud on the lower index, device on the higher, port 32300 + lower; flock /tmp/gpu<lo>.lock THEN
# /tmp/gpu<hi>.lock, run_ours.sh %(set)s %(arm)s G CHUNKJSON (it sees both locks held by its ancestors). Idempotent (runs/ours_%(arm)s_dev_%(set)s).
G=${1:?usage: $0 GPU}; B=/home/hk28965/specwm/baselines; NAME=%(stem)s
case "$G" in 0|1|2|3) ;; *) echo "[queue] $NAME: GPU $G: tier-9 hybrids run only on the pairs 0+1 and 2+3 (GPUs 4-7 not ours)"; exit 1;; esac
PG=$((G ^ 1)); LO=$(( G < PG ? G : PG )); HI=$(( G < PG ? PG : G ))
%(gate_sh)s
%(gate_calls)s
CJ=$B/raw/ours_%(arm)s_%(set)s/chunkdef_$NAME.json; mkdir -p $B/raw/ours_%(arm)s_%(set)s
cat > $CJ.tmp <<'EOF_CHUNK'
%(js)s
EOF_CHUNK
mv -f $CJ.tmp $CJ
t0=$(date +%%s); echo "[queue $(date '+%%m/%%d %%H:%%M:%%S')] $NAME: %(n)d clips, waiting for /tmp/gpu$LO.lock then /tmp/gpu$HI.lock (claimed by gpu $G)"
flock /tmp/gpu$LO.lock flock /tmp/gpu$HI.lock env CUDA_VISIBLE_DEVICES=$LO,$HI $B/run_ours.sh %(set)s %(arm)s $G $CJ; RC=$?
echo "[queue $(date '+%%m/%%d %%H:%%M:%%S')] $NAME: rc $RC, wall $(( $(date +%%s) - t0 )) s (lock wait + hold; hold in HYB_HOLD)"%(chk)s
exit $RC
""" % dict(made=MADE, tier=TIER, arm=arm, set=d["set"], n=d["n"], res=d["res"], dur=" of 10 s" if smoke else " of 60 s", smk=" [SMOKE %s]" % smoke if smoke else "", gates=",".join(gates) or "-",
           ids=",".join(d["ids"]), stem=stem, gate_sh=GATE_SH, gate_calls=gate_calls(gates), js=js, chk=chk)
def split(lst, k):
    n = len(lst); return [lst[i * n // k:(i + 1) * n // k] for i in range(k)] if n else []
def main():
    for sub in (str(TIER), "claimed", "done", "failed"): os.makedirs(os.path.join(Q, sub), exist_ok=True)
    os.makedirs(GEN + "/sentinels", exist_ok=True)
    queued = set()   # (arm, set, id) already in a chunk of this maker (pending / claimed / done)
    for f in glob.glob("%s/%d/*.sh" % (Q, TIER)) + glob.glob("%s/claimed/*.sh.gpu*" % Q) + glob.glob("%s/done/*.sh.gpu*" % Q):
        try:
            head = open(f).read(700)
            if "generated by gen/make_queue_matrix.py" not in head: continue
            for ln in head.splitlines():
                if ln.startswith(("# HYB_CHUNK ", "# OURS_CHUNK ")):
                    kv = dict(x.split("=", 1) for x in ln.split()[2:] if "=" in x); queued.update((kv.get("arm"), kv.get("set"), i) for i in kv.get("ids", "").split(",") if i); break
        except Exception: continue
    used = [int(os.path.basename(f).split("_")[0]) for sub in [d for d in os.listdir(Q) if d.isdigit()] + ["claimed", "done", "failed"] + [p for p in glob.glob(Q + "/park/*") if os.path.isdir(p)]
            for f in glob.glob("%s/%s/*" % (Q, sub)) if os.path.basename(f).split("_")[0].isdigit()]
    idx = max([a.start] + [u + 1 for u in used if u >= a.start]); assert idx == a.start or not a.dry_run or True
    T = {s: json.load(open("%s/tasks/%s.json" % (B, s))) for s in SETS + list(SMOKE_SET.values())}
    plan = []
    def emit(kind, arm, tset, tasks, gates, smoke=None, est=None, load=None):
        nonlocal idx
        tasks = [x for x in tasks if (arm, tset, x["id"]) not in queued]
        if not tasks: return
        name = "%03d_%s.sh" % (idx, ("smoke_%s" % smoke) if smoke else "%s_%s" % (tset, arm)); idx += 1
        d = dict(set=tset, arm=arm, tier=TIER, res=tasks[0]["res"], area=tasks[0]["area"], n=len(tasks), ids=[x["id"] for x in tasks], made=MADE, tasks=tasks, gates=gates)
        sc = hyb_script(name, d, gates, smoke) if kind == "hyb" else ours_script(name, d, gates)
        sec = (load or 0) + len(tasks) * (est or 0); gpus = 2 if kind == "hyb" else 1
        plan.append(dict(tier=TIER, name=name, kind=kind, arm=arm, set=tset, n=len(tasks), gpus=gpus, est_s=sec, gates=gates, smoke=smoke, res=d["res"]))
        if not a.dry_run:
            p = "%s/%d/%s" % (Q, TIER, name); tmp = p + ".tmp"; open(tmp, "w").write(sc); os.chmod(tmp, 0o755); os.replace(tmp, p)
    # ---- smokes: rt2c1hyb24 on one 10-s clip per resolution, gated on the tier-8 memory check of that resolution ----
    for tset in SETS:
        ss = SMOKE_SET[tset]; t = T[ss][0]; res = t["res"]
        emit("hyb", "rt2c1hyb24", ss, [t], ["mem24_%s" % res], smoke="rt2c1hyb24_%s" % ss, est=EST10["rt2c1hyb24"][RESI[res]], load=LOAD10["rt2c1hyb24"])
    # ---- the 60-s matrix ----
    ARMS = ["rt2c1hyb24", "rt2c1hyb32", "mx24", "mx32", "mxQ4", "mxQ4c1hyb", "mxK8", "mxK8c1hyb", "mxW12", "mxW12c1hyb", "mxW9", "mxW9c1hyb"] + (["mx32c1hyb", "mx24c1hyb"] if a.dup_hyb else [])
    for tset in SETS:
        res = T[tset][0]["res"]; ri = RESI[res]; smk = "rt2c1hyb24_%s" % SMOKE_SET[tset]
        for arm in ARMS:
            hyb = arm.endswith("c1hyb") or arm.startswith("rt2c1hyb"); fz = "%s/runs/ours_%s_%s%s" % (B, arm, "dev_" if hyb else "", tset)
            todo = [x for x in T[tset] if not os.path.exists("%s/%s/timing.json" % (fz, x["id"]))]
            if not todo: continue
            gates = [smk] + (["gptq4_save"] if ("24" in arm or "Q4" in arm) else [])
            est, load = EST[arm][ri], LOAD[arm]; per = max(1, min(CAP, int((TARGET - load) // est))); k = max(1, math.ceil(len(todo) / per))
            for part in split(todo, k): emit("hyb" if hyb else "ours", arm, tset, part, gates, None, est, load)
    # ---- report ----
    ph = sh = 0.0
    for r in plan:
        h = r["est_s"] / 3600.0
        if r["gpus"] == 2: ph += h
        else: sh += h
        print("tier %d %-40s %-5s %-12s %-17s %d clips %s  est %5.0f s (%4.2f h x %d GPU)  gates %s" % (r["tier"], r["name"], r["kind"], r["arm"], r["set"], r["n"], r["res"], r["est_s"], h, r["gpus"], ",".join(r["gates"]) or "-"))
    g = 2 * ph + sh
    print("[plan] tier %d: %d chunks, %d clips (%d smoke), est %.1f pair-hours + %.1f single-GPU-hours = %.1f GPU-hours (= %.1f h of the 4 worker GPUs if perfectly packed)%s"
          % (TIER, len(plan), sum(r["n"] for r in plan), sum(r["n"] for r in plan if r["smoke"]), ph, sh, g, g / 4, " (DRY RUN)" if a.dry_run else ""))
    if not a.dry_run and plan:
        with open(Q + "/INDEX.tsv", "a") as f:
            f.write("# tier 9 [10/19 thesis: memory-squeeze matrix on the 60-s P6 sets], %s: gen/make_queue_matrix.py wrote %d chunks (%d clips, est %.1f pair-hours + %.1f single-GPU-hours = %.1f GPU-hours): %s = rt2c1hyb24 10-s smokes (gated on gen/sentinels/mem24_<res>.ok from tier-8 311, itself gated on gptq4_save.ok from 310) -> gen/sentinels/rt2c1hyb24_<set>.ok|.fail; then the matrix arms %s on %s, each gated on the smoke of its resolution (+ gptq4_save for the gptq4 arms); mx32c1hyb / mx24c1hyb not queued (= rt2c1hyb32 / rt2c1hyb24); plan_matrix_1019.json\n"
                    % (MADE, len(plan), sum(r["n"] for r in plan), ph, sh, g, ",".join(r["name"] for r in plan if r["smoke"]), ",".join(ARMS), ",".join(SETS)))
        json.dump(dict(made=MADE, tier=TIER, est=EST, load=LOAD, target_s=TARGET, chunks=plan, pair_hours=round(ph, 2), single_gpu_hours=round(sh, 2), gpu_hours=round(g, 2)), open(Q + "/plan_matrix_1019.json", "w"), indent=1)
    return 0
if __name__ == "__main__": sys.exit(main())
