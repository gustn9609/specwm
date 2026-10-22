#!/usr/bin/env python3
"""[10/19 EXP 2 light device] Queue tier 7: the rt2c1hyb32 arm (= rt2c1hyb with a 32-GiB device: two_machine --window 12 --kv fp8 --cap 32 --lowmem 1 on the
DEVICE only; the cloud is rt2c1hyb's; run_ours.sh rt2c1hyb32; PROTOCOL §7) on the 10-s sets whose resolution FITS the 32-GiB cap (measured 10/19 on GPU 5
with a CPU stand-in cloud; the sets that do not fit are reported, not queued). Run automatically by the gen/worker.sh workers once tiers <= 6 have
nothing left to claim (tier dirs in numeric order; .sh chunks as `bash FILE G`, own process group, CHUNK_TIMEOUT 9000 s).
[10/19 PM v2] the device decodes spec.mp4 with --vae-dtype bf16 --vae-tiles 2,2 (run_ours.sh rt2c1hyb32), so 832x480 and 1024x512 fit too; ONE smoke per
resolution (--smoke-sets, default mg2,neutral_832,wp; astra gates on the 832x480 smoke of neutral_832).
Tier 7 = first the smoke chunks (NNN_smoke_rt2c1hyb32_<set>.sh, one clip each): pair run, then gen/smoke_check.py (--kind hyb: both streams, 161
frames, every correction received, >= 1 applied) AND a light-device check (the device timing.json has "mem" with cap_GiB 32, kv fp8, window 12, lowmem 1,
the expected --vae-dtype / --vae-tiles, reserved <= 32 GiB, device process AND device-GPU (HYB_HOLD) nvidia-smi peak <= --nvsmi-max MiB) ->
gen/sentinels/rt2c1hyb32_<set>.ok | .fail; then the full chunks, each FIRST gating on the sentinel of its resolution exactly
as tier 6 (gate(): .ok -> go; .fail -> exit 1; neither and the smoke chunk neither pending in queue/7 nor running in queue/claimed -> exit 1; else wait
<= GATE_WAIT s (default 3600) WITHOUT holding a GPU lock). Pair mechanics = tier 6 (flock gpu<lo> then gpu<hi>; cloud on lo, device on hi; GPUs 0-3 only).
Chunk sizing = tier 6's rule: load + n x clip <= TARGET 3000 s (per-clip / load estimates below), balanced split.
usage: python3 /home/hk28965/specwm/baselines/gen/make_queue_light.py --sets neutral_832,mg2,... [--smoke-sets mg2,neutral_832,wp] [--dry-run]   (plain python3;
re-running adds only clips not yet queued / frozen)"""
import os, sys, json, math, time, glob, argparse
B = "/home/hk28965/specwm/baselines"; GEN = B + "/gen"; sys.path.insert(0, GEN)
import make_queue_ours as MQ          # load_tasks (FROZEN order)
ARM = "rt2c1hyb32"; TIER = 7
ap = argparse.ArgumentParser(); ap.add_argument("--sets", required=True, help="comma list of 10-s sets that FIT the 32-GiB device (measured)"); ap.add_argument("--smoke-sets", default="mg2,neutral_832,wp", help="one smoke per resolution; a set gates on the smoke set with its resolution")
ap.add_argument("--vae-dtype", default="bf16"); ap.add_argument("--vae-tiles", default="2,2", help="the device decode run_ours.sh passes for rt2c1hyb32 (checked by the smoke)")
ap.add_argument("--nvsmi-max", type=int, default=32607, help="smoke: max device-process nvidia-smi MiB (32607 = an RTX 5090's nvidia-smi total)"); ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--queue-root", default=GEN + "/queue"); a = ap.parse_args()
Q = a.queue_root; MADE = time.strftime("%Y-%m-%d %H:%M:%S"); RES = ["1024x512", "640x352", "1280x704", "832x480"]
# per-clip seconds inside a warm pair hold (1024x512 / 640x352 / 1280x704 / 832x480) and load s per chunk process: rt2c1hyb's tier-6 estimates (68 / 27 / 125 / 51,
# load 520) + 10 % for the frame-by-frame decode and the FP8-KV reads, + 60 s for the CPU load + block-wise FP8 swap (10/19 GPU-5 device runs: see PROTOCOL §7)
EST = tuple(round(x * 1.10) for x in (68, 27, 125, 51)); LOAD = 580; TARGET = 3000
GATE_SH = r'''SENT=$B/gen/sentinels; QR=$B/gen/queue; GATE_WAIT=${GATE_WAIT:-3600}; mkdir -p $SENT
gate() {   # $1 = smoke name: go on when $SENT/$1.ok exists; exit 1 at once on $SENT/$1.fail or when the smoke chunk is neither pending (queue/7) nor running; else wait (no lock held)
  local n=$1 t0=$(date +%s)
  while :; do
    [ -f $SENT/$n.ok ] && { echo "[gate] smoke $n ok"; return 0; }
    [ -f $SENT/$n.fail ] && { echo "[gate] smoke $n FAILED: $(head -c 400 $SENT/$n.fail | tr '\n' ' ') -> exit 1"; exit 1; }
    # [10/19 fix] the former `ls A B || exit` exited 2 whenever ONE glob was unmatched (= the normal pending-or-running case)
    [ -n "$(ls $QR/7/*_smoke_$n.sh $QR/claimed/*_smoke_$n.sh.gpu* 2>/dev/null)" ] || { echo "[gate] smoke $n: no sentinel and its chunk is neither pending nor running (failed / killed?) -> exit 1"; exit 1; }
    [ $(( $(date +%s) - t0 )) -ge $GATE_WAIT ] && { echo "[gate] smoke $n still pending/running after ${GATE_WAIT} s -> exit 1 (re-queue this chunk once the smoke is done)"; exit 1; }
    sleep 30
  done
}'''
LIGHT_CHECK = r'''
if [ $RC -eq 0 ]; then   # light-device check on top of smoke_check.py: the device really ran capped (32 GiB, FP8 KV, window 12) and stayed below NVSMI_MAX
  python3 - "$B/runs/ours_rt2c1hyb32_dev_@SET@" "@IDS@" "$SENT/@SMOKE@" @NVMAX@ "$B/logs/ours_rt2c1hyb32_@SET@.log" <<'EOF_LIGHT'; RC=$?
import json, os, re, sys
runs, ids, sent, nvmax, hlog = sys.argv[1], sys.argv[2].split(","), sys.argv[3], int(sys.argv[4]), sys.argv[5]; bad = []; got = {}
hold = [ln for ln in open(hlog) if ln.startswith("HYB_HOLD ")] if os.path.exists(hlog) else []   # run_ours.sh: GPU-wide nvidia-smi peaks of the pair over the hold (2-s polling)
m = re.search(r"device gpu (\d+):.*nvidia-smi peak (.*), rc", hold[-1]) if hold else None
dpk = int(re.search(r"gpu%s (\d+) MiB" % m.group(1), m.group(2)).group(1)) if m and re.search(r"gpu%s (\d+) MiB" % m.group(1), m.group(2)) else None
got["hyb_hold_device_gpu_peak_MiB"] = dpk
if dpk is None or dpk > nvmax: bad.append("HYB_HOLD device-GPU nvidia-smi peak %s MiB (whole GPU, whole hold) > %d or missing" % (dpk, nvmax))
for i in ids:
    T = json.load(open(os.path.join(runs, i, "timing.json"))); M = T.get("device_mem") or {}
    got[i] = dict(cap=M.get("cap_GiB"), kv=M.get("kv"), window=M.get("window"), lowmem=M.get("lowmem"), proc_peak_alloc_GiB=M.get("proc_peak_alloc_GiB"), proc_peak_reserved_GiB=M.get("proc_peak_reserved_GiB"), nvsmi_peak_MiB=M.get("nvsmi_proc_peak_MiB"), kv_window=T.get("kv_window"))
    if not M: bad.append("%s: device timing has no mem record" % i); continue
    got[i].update(vae_dtype=M.get("vae_dtype"), vae_tiles=M.get("vae_tiles"))
    if M.get("cap_GiB") != 32.0 or M.get("kv") != "fp8" or M.get("window") != 12 or M.get("lowmem") != 1 or M.get("vae_dtype") != "@VAE_DTYPE@" or M.get("vae_tiles") != "@VAE_TILES@": bad.append("%s: device flags %s" % (i, got[i]))
    if (M.get("proc_peak_reserved_GiB") or 99) > 32.0: bad.append("%s: device reserved peak %s GiB > 32" % (i, M.get("proc_peak_reserved_GiB")))
    if M.get("nvsmi_proc_peak_MiB") is None or M["nvsmi_proc_peak_MiB"] > nvmax: bad.append("%s: device nvidia-smi peak %s MiB > %d" % (i, M.get("nvsmi_proc_peak_MiB"), nvmax))
ok = sent + ".ok"; fail = sent + ".fail"
if bad:
    d = json.load(open(ok)) if os.path.exists(ok) else {}; d.update(ok=False, light_problems=bad, light=got)
    json.dump(d, open(fail, "w"), indent=1); os.path.exists(ok) and os.remove(ok); print("[light] FAIL", bad); sys.exit(1)
d = json.load(open(ok)); d["light"] = got; json.dump(d, open(ok + ".tmp", "w"), indent=1); os.replace(ok + ".tmp", ok); print("[light] ok", got); sys.exit(0)
EOF_LIGHT
fi'''
def hyb_script(name, d, gates, smoke=None):
    """GPU-pair chunk (= make_queue_long.hyb_script, tier 7); HYB_CHUNK line kept for make_queue_ours.queued_ids"""
    stem = name[:-3]; js = json.dumps(d, indent=1); assert "\nEOF_CHUNK\n" not in js
    chk = ""
    if smoke:
        chk = ("\npython3 $B/gen/smoke_check.py %s --kind hyb --runs $B/runs/ours_%s_dev_%s --runs-cloud $B/runs/ours_%s_cloud_%s --ids %s --frames 161 --rc $RC --note \"$NAME (see HYB_HOLD in logs/ours_%s_%s.log)\"; RC=$?"
               % (smoke, ARM, d["set"], ARM, d["set"], ",".join(d["ids"]), ARM, d["set"])) + LIGHT_CHECK.replace("@SET@", d["set"]).replace("@IDS@", ",".join(d["ids"])).replace("@NVMAX@", str(a.nvsmi_max)).replace("@SMOKE@", smoke).replace("@VAE_DTYPE@", a.vae_dtype).replace("@VAE_TILES@", a.vae_tiles)
    return """#!/bin/bash
# generated by gen/make_queue_light.py %(made)s -- tier %(tier)d: OUR hybrid %(arm)s (32-GiB light device) on %(set)s, %(n)d clips at %(res)s, on a GPU PAIR%(smk)s (re-run make_queue_light.py, do not edit)
# HYB_CHUNK set=%(set)s arm=%(arm)s n=%(n)d gates=%(gates)s ids=%(ids)s
# usage: bash THIS GPU   (worker.sh runs .sh chunks so, WITHOUT a lock): gate on the tier-7 smoke sentinel first (no lock held while waiting), then the tier-6
# pair mechanics: pair of G = G xor 1 (ONLY 0+1 and 2+3: GPUs 4-7 are not ours; G 4-7 exits 1), cloud on the lower index, device on the higher, port 32300 + lower; flock /tmp/gpu<lo>.lock THEN
# /tmp/gpu<hi>.lock, run_ours.sh %(set)s %(arm)s G CHUNKJSON (it sees both locks held by its ancestors). Idempotent (runs/ours_%(arm)s_dev_%(set)s).
G=${1:?usage: $0 GPU}; B=/home/hk28965/specwm/baselines; NAME=%(stem)s
case "$G" in 0|1|2|3) ;; *) echo "[queue] $NAME: GPU $G: tier-7 hybrids run only on the pairs 0+1 and 2+3 (GPUs 4-7 not ours)"; exit 1;; esac
PG=$((G ^ 1)); LO=$(( G < PG ? G : PG )); HI=$(( G < PG ? PG : G ))
SENT=$B/gen/sentinels; mkdir -p $SENT
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
""" % dict(made=MADE, tier=TIER, arm=ARM, set=d["set"], n=d["n"], res=d["res"], smk=" [SMOKE %s]" % smoke if smoke else "", gates=",".join(gates) or "-",
           ids=",".join(d["ids"]), stem=stem, gate_sh=GATE_SH if gates else "", gate_calls="\n".join("gate %s" % g for g in gates), js=js, chk=chk)
def split(lst, k):
    n = len(lst); return [lst[i * n // k:(i + 1) * n // k] for i in range(k)] if n else []
def main():
    sets = [s for s in a.sets.split(",") if s]; smokes = [s for s in a.smoke_sets.split(",") if s]; assert all(s in sets for s in smokes), "every smoke set must be one of the fitting sets"
    for sub in (str(TIER), "claimed", "done", "failed"): os.makedirs(os.path.join(Q, sub), exist_ok=True)
    os.makedirs(GEN + "/sentinels", exist_ok=True)
    queued = set()   # (set, id) already inside a tier-7 chunk of this maker (pending / claimed / done)
    for f in glob.glob("%s/%d/*.sh" % (Q, TIER)) + glob.glob("%s/claimed/*.sh.gpu*" % Q) + glob.glob("%s/done/*.sh.gpu*" % Q):
        try:
            head = open(f).read(600)
            if "generated by gen/make_queue_light.py" not in head: continue
            for ln in head.splitlines():
                if ln.startswith("# HYB_CHUNK "):
                    kv = dict(x.split("=", 1) for x in ln.split()[2:] if "=" in x)
                    if kv.get("arm") == ARM: queued.update((kv.get("set"), i) for i in kv.get("ids", "").split(",") if i)
                    break
        except Exception: continue
    idx = 1 + max([int(os.path.basename(f).split("_")[0]) for sub in [d for d in os.listdir(Q) if d.isdigit()] + ["claimed", "done", "failed"]
                   for f in glob.glob("%s/%s/*" % (Q, sub)) if os.path.basename(f).split("_")[0].isdigit()] or [-1])   # after every NNN in use (all tiers)
    plan = []; T = {}
    for s in sets:
        t, _ = MQ.load_tasks(s, False); assert t, s; T[s] = t
    def emit(tset, tasks, gates, smoke=None):
        nonlocal idx
        tasks = [x for x in tasks if (tset, x["id"]) not in queued]
        if not tasks: return
        name = "%03d_%s.sh" % (idx, ("smoke_%s" % smoke) if smoke else "%s_%s" % (tset, ARM)); idx += 1
        d = dict(set=tset, arm=ARM, tier=TIER, res=tasks[0]["res"], area=tasks[0]["area"], n=len(tasks), ids=[x["id"] for x in tasks], made=MADE, tasks=tasks, gates=gates)
        est = EST[RES.index(tasks[0]["res"])]; sec = LOAD + len(tasks) * est
        plan.append(dict(tier=TIER, name=name, kind="hyb", arm=ARM, set=tset, n=len(tasks), gpus=2, est_s=sec, gates=gates, smoke=smoke, vae_dtype=a.vae_dtype, vae_tiles=a.vae_tiles))
        if not a.dry_run:
            p = "%s/%d/%s" % (Q, TIER, name); tmp = p + ".tmp"; open(tmp, "w").write(hyb_script(name, d, gates, smoke)); os.chmod(tmp, 0o755); os.replace(tmp, p)
    sm_of = {}   # set -> smoke name of its resolution
    for ss in smokes:
        emit(ss, [T[ss][0]], [], smoke="%s_%s" % (ARM, ss))
        for tset in sets:
            if T[tset][0]["res"] == T[ss][0]["res"]: sm_of.setdefault(tset, "%s_%s" % (ARM, ss))
    assert all(t in sm_of for t in sets), ("a set without a smoke of its resolution", [t for t in sets if t not in sm_of])
    sm_ids = {T[ss][0]["id"] for ss in smokes}
    for tset in sets:
        todo = [x for x in T[tset] if x["id"] not in sm_ids and not os.path.exists("%s/runs/ours_%s_dev_%s/%s/timing.json" % (B, ARM, tset, x["id"]))]
        est = EST[RES.index(T[tset][0]["res"])]; per = max(1, int((TARGET - LOAD) // est)); k = max(1, math.ceil(len(todo) / per))
        for part in split(todo, k): emit(tset, part, [sm_of[tset]])
    ph = 0.0
    for r in plan:
        h = r["est_s"] / 3600.0; ph += h
        print("tier %d %-36s %-11s %-12s %3d clips  est %5.0f s (%4.2f pair-h)  gates %s" % (r["tier"], r["name"], r["arm"], r["set"], r["n"], r["est_s"], h, ",".join(r["gates"]) or "-"))
    print("[plan] tier %d: %d chunks, %d clips, est %.1f pair-hours = %.1f GPU-hours%s" % (TIER, len(plan), sum(r["n"] for r in plan), ph, 2 * ph, " (DRY RUN)" if a.dry_run else ""))
    if not a.dry_run and plan:
        with open(Q + "/INDEX.tsv", "a") as f:
            f.write("# tier 7 v2 [10/19 PM EXP 2 light device rt2c1hyb32, device decode --vae-dtype %s --vae-tiles %s], %s: gen/make_queue_light.py wrote %d chunks (%d clips, est %.1f pair-hours) on sets %s (the sets whose resolution fits the 32-GiB device cap); smokes %s -> gen/sentinels/rt2c1hyb32_<set>.ok|.fail (smoke_check + light-device check), each set gates on the smoke of its resolution; replaces the v1 tier-7 chunks 277-279 (never claimed; deleted); runs/ours_rt2c1hyb32_{dev,cloud}_<set>\n"
                    % (a.vae_dtype, a.vae_tiles, MADE, len(plan), sum(r["n"] for r in plan), ph, ",".join(sets), ",".join(r["name"] for r in plan if r["smoke"])))
        json.dump(plan, open(Q + "/plan_light_1019.json", "w"), indent=1)
    return 0
if __name__ == "__main__": sys.exit(main())
