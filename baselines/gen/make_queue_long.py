#!/usr/bin/env python3
"""[10/19] Queue tiers 5 + 6 for three new experiments; run AUTOMATICALLY by the six gen/worker.sh workers once tier 4 has nothing left to claim
(the workers take tier dirs in numeric order, and .sh chunks as `bash FILE G`, own process group, CHUNK_TIMEOUT 9000 s).
  EXP 1  long rollouts: the 60-s program P6 (baselines/convert.py, programs/P6.npz) on 4 frozen scenes of neutral_832 / mg2 / abot / wp
         (task sets <set>_long = tasks/<set>_long.json, ids <set>__<scene>__P6_long, 969 poses -> 961 frames): the set's own baseline (Waypoint / MG2 / ABot,
         native setting, 60 s at native fps) + ours teacher, rt2, adapt25, sf24, adapt25c1hyb, rt2c1hyb, sf24c1hyb.
  EXP 2  rt2c1hyb  (light device: rt2 + cloud 1 step at t=750 + KV feedback) on neutral_832, wp, mg2, abot, astra (7 programs, all frozen scenes) + P6.
  EXP 3  sf24c1hyb (device = Speculative Forcing on the 14B, token-wise early exit, + cloud 1 step at t=750 + KV feedback), same matrix as EXP 2.
  (rt2c1hyb32, the 32-GB device with FP8 KV + window 12, is NOT queued: two_machine has no FP8-KV option.)
Tier 5 = one smoke clip per new path; each smoke chunk runs its clip (skip-if-done: a clip already frozen is just re-checked) and then
gen/smoke_check.py writes gen/sentinels/<smoke>.ok (or .fail). Tier 6 = the full runs; every tier-6 chunk script FIRST calls gate <smoke>... for the
smokes its path depends on: <smoke>.ok -> go on; <smoke>.fail -> exit 1 at once; no sentinel and the smoke chunk neither pending (queue/5) nor running
(queue/claimed) -> exit 1 at once; smoke still pending/running -> wait for it (poll 30 s, at most GATE_WAIT s, default 3600; GATE_WAIT=0 = never wait),
WITHOUT holding any GPU lock. So a broken path costs seconds, and a tier-6 chunk claimed while its smoke is still running does not fail spuriously.
Chunks: our single-GPU arms = flock /tmp/gpu<G>.lock + _gpu_peak_run.sh + run_ours.sh SET ARM G CHUNKJSON; our hybrids = the tier-4 pair mechanics
(flock gpu<lo> then gpu<hi>, run_ours.sh SET ARM G CHUNKJSON, cloud on lo / device on hi); baselines = their batch runners under flock gpu<G> (jobs file
written by the chunk itself, gen/filter_pending.py first). Every chunk is idempotent (frozen clips are skipped) and sized from the per-clip estimates
below so that load + n x clip <= TARGET (3000 s for 10-s hybrid chunks, 4500 s for P6 chunks; CHUNK_TIMEOUT is 9000 s).
Order: tier 5 = rt2c1hyb, sf24c1hyb (pair smokes first, see SMOKES), p6_teacher832, p6_waypoint, p6_mg2, p6_abot, p6_sf24_832, p6_teacher1280; tier 6 = EXP 3 + EXP 2 on neutral_832,
then their other sets (EXP 3 before EXP 2 per set), then EXP 1 (P6 baselines, then ours per set neutral_832_long, mg2_long, wp_long, abot_long).
usage: python3 /home/hk28965/specwm/baselines/gen/make_queue_long.py [--dry-run] [--queue-root DIR]      (plain python3; re-run only to add missing chunks)"""
import os, sys, json, math, time, glob, argparse
B = "/home/hk28965/specwm/baselines"; GEN = B + "/gen"; sys.path.insert(0, GEN)
import make_queue_ours as MQ          # load_tasks (FROZEN order), next_index, queued_ids
import frozen_scenes as FS
ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--queue-root", default=GEN + "/queue"); a = ap.parse_args()
Q = a.queue_root; MADE = time.strftime("%Y-%m-%d %H:%M:%S"); RES = ["1024x512", "640x352", "1280x704", "832x480"]
SETS10 = ["neutral_832", "wp", "mg2", "abot", "astra"]; SETS_LONG = ["neutral_832_long", "mg2_long", "wp_long", "abot_long"]
HYB = ("adapt25c1hyb", "rt2c1hyb", "sf24c1hyb")
# ---- per-clip estimates (s, one clip inside a warm hold) by resolution 1024x512 / 640x352 / 1280x704 / 832x480 and load (s per chunk process) ----
# 10-s hybrids: tier-4 HYB_HOLD fits (adapt25c1hyb: 68 / 27 / 125 / 51 s per clip + ~510 s load); rt2c1hyb ~ the same; sf24c1hyb = device sf24 gen x 1.35
# (transport) + the same overhead (74 / 86 / 34 / 155 s) and +150 s of dynamic-shape compile in the device warm-up.
EST10 = {"rt2c1hyb": (68, 27, 125, 51), "sf24c1hyb": (86, 34, 155, 74)}; LOAD10 = {"rt2c1hyb": 520, "sf24c1hyb": 670}
# 60-s P6 = x 5.8 the 10-s clip (81 vs 14 chunks; prepare / decode / conversion scale with the frames too); two_machine warms up on ALL 81 chunk positions
# (static compile per position) -> loads are larger; spec_early warms on a 10-s clip (--warm-frames 165). [10/19 GPU-6 smoke numbers replace these when known]
ESTL = {"teacher": (586, 238, 1021, 435), "rt2": (371, 145, 626, 273), "adapt25": (386, 151, 651, 284), "sf24": (348, 133, 638, 297),
        "adapt25c1hyb": (394, 157, 725, 296), "rt2c1hyb": (394, 157, 725, 296), "sf24c1hyb": (499, 197, 899, 429)}
LOADL = {"teacher": 900, "rt2": 800, "adapt25": 800, "sf24": 400, "adapt25c1hyb": 1300, "rt2c1hyb": 1300, "sf24c1hyb": 1500}
BASEL = {"waypoint": (130, 240), "mg2": (40, 90), "abot": (70, 140)}   # P6 baseline (per clip, load+warm-up) s
TARGET10, TARGETL, CAPL = 3000, 4500, 4
EST_FILE = GEN + "/queue/long_estimates.json"      # optional override written after the smokes ({"ESTL": {...}, "LOADL": {...}})
if os.path.exists(EST_FILE):
    o = json.load(open(EST_FILE)); ESTL.update({k: tuple(v) for k, v in o.get("ESTL", {}).items()}); LOADL.update(o.get("LOADL", {}))
SMOKES = [  # name, kind, arm, set, id, frames.  [10/19] the two PAIR smokes first: with only two pairs (0+1, 2+3) they gate the first tier-6 chunks, so they
    # must not queue behind the long single-GPU smokes; the 1280x704 P6 smoke (the longest, gates only the last tier-6 chunks) is last
    ("rt2c1hyb", "hyb", "rt2c1hyb", "neutral_832", None, 161), ("sf24c1hyb", "hyb", "sf24c1hyb", "neutral_832", None, 161),
    ("p6_teacher832", "ours", "teacher", "neutral_832_long", None, 961), ("p6_waypoint", "base", "waypoint", "wp_long", None, 3600),
    ("p6_mg2", "base", "mg2", "mg2_long", None, 720), ("p6_abot", "base", "abot", "abot_long", None, 720),
    ("p6_sf24_832", "ours", "sf24", "neutral_832_long", None, 961), ("p6_teacher1280", "ours", "teacher", "abot_long", None, 961)]
def gates_for(arm, tset):
    if not tset.endswith("_long"): return [arm]                                    # EXP 2 / 3 10-s matrix: the arm's own pair smoke
    g = {"teacher": ["p6_teacher832"], "rt2": ["p6_teacher832"], "adapt25": ["p6_teacher832"], "sf24": ["p6_sf24_832"], "adapt25c1hyb": ["p6_teacher832"],
         "rt2c1hyb": ["p6_teacher832", "rt2c1hyb"], "sf24c1hyb": ["p6_teacher832", "sf24c1hyb"], "waypoint": ["p6_waypoint"], "mg2": ["p6_mg2"], "abot": ["p6_abot"]}[arm]
    return g + (["p6_teacher1280"] if tset == "abot_long" and arm not in ("waypoint", "mg2", "abot") else [])
GATE_SH = r'''SENT=$B/gen/sentinels; QR=$B/gen/queue; GATE_WAIT=${GATE_WAIT:-3600}; mkdir -p $SENT
gate() {   # $1 = smoke name: go on when $SENT/$1.ok exists; exit 1 at once on $SENT/$1.fail or when the smoke chunk is neither pending nor running; else wait (no lock held)
  local n=$1 t0=$(date +%s)
  while :; do
    [ -f $SENT/$n.ok ] && { echo "[gate] smoke $n ok"; return 0; }
    [ -f $SENT/$n.fail ] && { echo "[gate] smoke $n FAILED: $(head -c 400 $SENT/$n.fail | tr '\n' ' ') -> exit 1"; exit 1; }
    ls $QR/5/*_smoke_$n.sh $QR/claimed/*_smoke_$n.sh.gpu* > /dev/null 2>&1 || { echo "[gate] smoke $n: no sentinel and its chunk is neither pending nor running (failed / killed?) -> exit 1"; exit 1; }
    [ $(( $(date +%s) - t0 )) -ge $GATE_WAIT ] && { echo "[gate] smoke $n still pending/running after ${GATE_WAIT} s -> exit 1 (re-queue this chunk once the smoke is done)"; exit 1; }
    sleep 30
  done
}'''
def ours_script(name, tier, d, gates, smoke=None):
    """single-GPU OUR arm (two_machine solo / spec_early) chunk"""
    stem = name[:-3]; js = json.dumps(d, indent=1); assert "\nEOF_CHUNK\n" not in js
    chk = ("\npython3 $B/gen/smoke_check.py %s --kind ours --runs $B/runs/ours_%s_%s --ids %s --frames %d --rc $RC --note \"$NAME hold ${HELD:-?} s peak ${PEAK:-?} MiB\"; RC=$?"
           % (smoke[0], d["arm"], d["set"], ",".join(d["ids"]), smoke[1])) if smoke else ""
    return """#!/bin/bash
# generated by gen/make_queue_long.py %(made)s -- tier %(tier)d: OUR %(arm)s on %(set)s, %(n)d clips at %(res)s, ONE GPU%(smk)s (re-run make_queue_long.py, do not edit)
# OURS_CHUNK set=%(set)s arm=%(arm)s n=%(n)d gates=%(gates)s ids=%(ids)s
# usage: bash THIS GPU   (worker.sh runs .sh chunks so, WITHOUT a lock): gate on the tier-5 smoke sentinels (none for a smoke itself), then
# flock /tmp/gpu$G.lock env CUDA_VISIBLE_DEVICES=$G _gpu_peak_run.sh run_ours.sh %(set)s %(arm)s $G CHUNKJSON (skip-if-done; FRAMES = the tasks' frames).
G=${1:?usage: $0 GPU}; B=/home/hk28965/specwm/baselines; NAME=%(stem)s
case "$G" in 0|1|2|3) ;; *) echo "[queue] $NAME: GPU $G: tiers 5/6 run only on GPUs 0-3 (10/19: GPUs 4-7 not ours)"; exit 1;; esac
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
echo "[queue $(date '+%%m/%%d %%H:%%M:%%S')] $NAME: rc $RC, wall $(( $(date +%%s) - t0 )) s, lock hold ${HELD:-?} s, peak GPU memory ${PEAK:-?} MiB"%(chk)s
exit $RC
""" % dict(made=MADE, tier=tier, arm=d["arm"], set=d["set"], n=d["n"], res=d["res"], smk=" [SMOKE %s]" % smoke[0] if smoke else "", gates=",".join(gates) or "-",
           ids=",".join(d["ids"]), stem=stem, gate_sh=GATE_SH if gates else "", gate_calls="\n".join("gate %s" % g for g in gates), js=js, chk=chk)
def hyb_script(name, tier, d, gates, smoke=None):
    """GPU-pair hybrid chunk (= make_queue_ours.hyb_script + gates + optional smoke check); HYB_CHUNK line kept for make_queue_ours.queued_ids"""
    stem = name[:-3]; js = json.dumps(d, indent=1); assert "\nEOF_CHUNK\n" not in js; arm = d["arm"]
    chk = ("\npython3 $B/gen/smoke_check.py %s --kind hyb --runs $B/runs/ours_%s_dev_%s --runs-cloud $B/runs/ours_%s_cloud_%s --ids %s --frames %d --rc $RC%s --note \"$NAME (see HYB_HOLD in logs/ours_%s_%s.log)\"; RC=$?"
           % (smoke[0], arm, d["set"], arm, d["set"], ",".join(d["ids"]), smoke[1], " --ee" if arm == "sf24c1hyb" else "", arm, d["set"])) if smoke else ""
    return """#!/bin/bash
# generated by gen/make_queue_long.py %(made)s -- tier %(tier)d: OUR hybrid %(arm)s on %(set)s, %(n)d clips at %(res)s, on a GPU PAIR%(smk)s (re-run make_queue_long.py, do not edit)
# HYB_CHUNK set=%(set)s arm=%(arm)s n=%(n)d gates=%(gates)s ids=%(ids)s
# usage: bash THIS GPU   (worker.sh runs .sh chunks so, WITHOUT a lock): gate on the tier-5 smoke sentinels first (no lock held while waiting), then the tier-4
# pair mechanics: pair of G = G xor 1 (ONLY 0+1 and 2+3 since 10/19: GPUs 4-7 are not ours; G 4-7 exits 1), cloud on the lower index, device on the higher, port 32300 + lower; flock /tmp/gpu<lo>.lock THEN
# /tmp/gpu<hi>.lock, run_ours.sh %(set)s %(arm)s G CHUNKJSON (it sees both locks held by its ancestors). Idempotent (runs/ours_%(arm)s_dev_%(set)s).
G=${1:?usage: $0 GPU}; B=/home/hk28965/specwm/baselines; NAME=%(stem)s
case "$G" in 0|1|2|3) ;; *) echo "[queue] $NAME: GPU $G: tiers 5/6 hybrids run only on the pairs 0+1 and 2+3 (10/19: GPUs 4-7 not ours)"; exit 1;; esac
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
""" % dict(made=MADE, tier=tier, arm=arm, set=d["set"], n=d["n"], res=d["res"], smk=" [SMOKE %s]" % smoke[0] if smoke else "", gates=",".join(gates) or "-",
           ids=",".join(d["ids"]), stem=stem, gate_sh=GATE_SH if gates else "", gate_calls="\n".join("gate %s" % g for g in gates), js=js, chk=chk)
def base_jobs(model, tasks):
    """baseline job list for P6 clips (out = runs/<model>/<set>__<scene>__P6_long); images = the set's own scene images (each runner's own loader)"""
    sset = {"waypoint": "wp", "mg2": "mg2", "abot": "abot"}[model]; jobs = []
    for t in tasks:
        img = FS.image_path(sset, t["scene"]); out = "%s/runs/%s/%s" % (B, model, t["id"])
        if model == "waypoint": jobs.append((img, t["program"], 1234, t["id"]))
        elif model == "mg2": jobs.append(dict(image=img, program=t["program"], seed=1234, out=out, set=sset, scene=t["scene"]))
        else:
            cap_f = os.path.splitext(img)[0] + ".txt"; cap = open(cap_f).read().strip() if os.path.exists(cap_f) else ""
            jobs.append(dict(image=img, program=t["program"], seed=1234, out=out, prompt=cap, scene=t["scene"], set=sset, actions=PROG_ABOT[t["program"]]))
    return jobs
PROG_ABOT = {}
def base_script(name, tier, model, tset, tasks, gates, smoke=None):
    stem = name[:-3]; jobs = base_jobs(model, tasks); n = len(jobs)
    if model == "waypoint":
        body = "# %s (%d clips): IMAGE\tPROGRAM\tSEED\tNAME -> runs/waypoint/NAME\n" % (stem, n) + "".join("%s\t%s\t%d\t%s\n" % r for r in jobs); ext = "tsv"
        cmd = 'env WP_JOBS=$J/%s/batch_jobs.json $B/waypoint/run.sh --batch "$PEND" $B/runs/waypoint' % stem; fx = " --out-root $B/runs/waypoint"
    else:
        body = json.dumps(jobs, indent=1) + "\n"; ext = "json"; fx = ""
        cmd = '$B/mg2/run.sh --batch "$PEND" --frames 729 --keep-frames 720' if model == "mg2" else '$B/abot/run.sh --batch "$PEND" --blocks 61 --keep-frames 720'
    assert "\nEOF_JOBS\n" not in body
    ids = [t["id"] for t in tasks]; frames = {"waypoint": 3600, "mg2": 720, "abot": 720}[model]
    chk = ("\npython3 $B/gen/smoke_check.py %s --kind base --runs $B/runs/%s --ids %s --frames %d --rc $RC --note \"$NAME hold ${HELD:-?} s peak ${PEAK:-?} MiB\"; RC=$?"
           % (smoke, model, ",".join(ids), frames)) if smoke else ""
    return """#!/bin/bash
# generated by gen/make_queue_long.py %(made)s -- tier %(tier)d: BASELINE %(model)s P6 (60 s, native setting) on %(set)s, %(n)d clips%(smk)s (re-run make_queue_long.py, do not edit)
# BASE_CHUNK model=%(model)s set=%(set)s n=%(n)d gates=%(gates)s ids=%(ids)s
# usage: bash THIS GPU: gate (tier-6 only), write the jobs file, keep the pending ones (gen/filter_pending.py), then flock /tmp/gpu$G.lock + _gpu_peak_run.sh
# + the runner's batch mode (%(cmd_short)s). P6 layouts (baselines/convert.py long_layout): Waypoint 900 calls = 3600 frames at 60 fps; MG2 729 frames generated,
# 720 written (12 fps); ABot 61 blocks = 729 frames, 720 written (12 fps), keys passed per job ("actions", = convert.encode_program('P6', 'abot')).
G=${1:?usage: $0 GPU}; S=/home/hk28965/specwm; B=$S/baselines; J=$B/gen/jobs; NAME=%(stem)s; L=$B/gen/logs/$NAME.log; mkdir -p $J $B/gen/logs
case "$G" in 0|1|2|3) ;; *) echo "[queue] $NAME: GPU $G: tiers 5/6 run only on GPUs 0-3 (10/19: GPUs 4-7 not ours)"; exit 1;; esac
%(gate_sh)s
%(gate_calls)s
SRC=$J/$NAME.%(ext)s; PEND=$J/$NAME.pending.%(ext)s
cat > $SRC.tmp <<'EOF_JOBS'
%(body)sEOF_JOBS
mv -f $SRC.tmp $SRC
N=$(python3 $B/gen/filter_pending.py "$SRC" "$PEND"%(fx)s) || { echo "[queue] filter_pending failed on $SRC" | tee -a $L; exit 2; }
echo "[queue $(date '+%%m/%%d %%H:%%M:%%S')] $NAME: $N of %(n)d clips pending (gpu $G)" | tee -a $L
RC=0; HELD=0; PEAK=0
if [ "$N" -gt 0 ]; then
  set -o pipefail; PK=$(mktemp); t0=$(date +%%s)
  flock /tmp/gpu$G.lock env CUDA_VISIBLE_DEVICES=$G $B/_gpu_peak_run.sh $G "$PK" %(cmd)s 2>&1 | tee -a $L; RC=${PIPESTATUS[0]}
  read -r HELD PEAK < "$PK" 2>/dev/null; rm -f "$PK"; set +o pipefail
  echo "[queue $(date '+%%m/%%d %%H:%%M:%%S')] $NAME: rc $RC, wall $(( $(date +%%s) - t0 )) s, hold ${HELD:-?} s on gpu $G, peak GPU memory ${PEAK:-?} MiB" | tee -a $L
fi%(chk)s
exit $RC
""" % dict(made=MADE, tier=tier, model=model, set=tset, n=n, smk=" [SMOKE %s]" % smoke if smoke else "", gates=",".join(gates) or "-", ids=",".join(ids), stem=stem,
           cmd_short=cmd.split()[0] if model != "waypoint" else "waypoint/run.sh --batch", gate_sh=GATE_SH if gates else "", gate_calls="\n".join("gate %s" % g for g in gates),
           ext=ext, body=body, fx=fx, cmd=cmd, chk=chk)
def split(lst, k):   # k balanced parts
    n = len(lst); return [lst[i * n // k:(i + 1) * n // k] for i in range(k)] if n else []
def main():
    sys.path.insert(0, B); import importlib; global PROG_ABOT
    # ABot P6 keys from the shared encoder (plain python3 has no numpy: read them from programs/P6.npz's JSON via a tiny envrun call)
    import subprocess
    r = subprocess.run(["/home/hk28965/specwm/scripts/envrun.sh", "python", "-c", "import numpy as np, json; z = np.load('%s/programs/P6.npz'); print(json.dumps(json.loads(str(z['abot']))['actions']))" % B], capture_output=True, text=True)
    PROG_ABOT = {"P6": json.loads(r.stdout.strip().splitlines()[-1])}; assert len(PROG_ABOT["P6"]) == 61, len(PROG_ABOT["P6"])
    for sub in ("5", "6", "claimed", "done", "failed"): os.makedirs(os.path.join(Q, sub), exist_ok=True)
    os.makedirs(GEN + "/sentinels", exist_ok=True)
    queued = set()   # (arm-or-model, set, id) already inside a pending / claimed / done tier-5/6 chunk of this maker (re-running adds only what is missing)
    for f in glob.glob("%s/5/*.sh" % Q) + glob.glob("%s/6/*.sh" % Q) + glob.glob("%s/claimed/*.sh.gpu*" % Q) + glob.glob("%s/done/*.sh.gpu*" % Q):
        try:
            for ln in open(f):
                if ln.startswith(("# HYB_CHUNK ", "# OURS_CHUNK ", "# BASE_CHUNK ")) and "generated by gen/make_queue_long.py" in open(f).read(400):
                    kv = dict(x.split("=", 1) for x in ln.split()[2:] if "=" in x); queued.update((kv.get("arm") or kv.get("model"), kv.get("set"), i) for i in kv.get("ids", "").split(",") if i)
                    break
        except Exception: continue
    idx = 1 + max([int(os.path.basename(f).split("_")[0]) for sub in [d for d in os.listdir(Q) if d.isdigit()] + ["claimed", "done", "failed"]
                   for f in glob.glob("%s/%s/*" % (Q, sub)) if os.path.basename(f).split("_")[0].isdigit()] or [-1])   # after every NNN in use (all tiers)
    plan = []; T = {}
    for s in SETS10 + SETS_LONG:
        t, _ = MQ.load_tasks(s, False) if not s.endswith("_long") else (json.load(open("%s/tasks/%s.json" % (B, s))), None)
        assert t, s; T[s] = t
    def res_i(s): return RES.index(T[s][0]["res"])
    def emit(tier, kind, arm, tset, tasks, gates, smoke=None, est=None, load=None):
        nonlocal idx
        tasks = [x for x in tasks if (arm, tset, x["id"]) not in queued]
        if not tasks: return
        base = ("smoke_%s" % smoke[0] if smoke and kind != "base" else ("smoke_%s" % smoke if smoke else "%s_%s" % (tset, arm))) + ".sh"
        name = "%03d_%s" % (idx, base); idx += 1
        d = dict(set=tset, arm=arm, tier=tier, res=tasks[0]["res"], area=tasks[0]["area"], n=len(tasks), ids=[x["id"] for x in tasks], made=MADE, tasks=tasks, gates=gates)
        sc = ours_script(name, tier, d, gates, smoke) if kind == "ours" else (hyb_script(name, tier, d, gates, smoke) if kind == "hyb" else base_script(name, tier, arm, tset, tasks, gates, smoke))
        p = "%s/%d/%s" % (Q, tier, name); gpus = 2 if kind == "hyb" else 1; sec = (load or 0) + len(tasks) * (est or 0)
        plan.append(dict(tier=tier, name=name, kind=kind, arm=arm, set=tset, n=len(tasks), gpus=gpus, est_s=sec, gates=gates))
        if not a.dry_run:
            tmp = p + ".tmp"; open(tmp, "w").write(sc); os.chmod(tmp, 0o755); os.replace(tmp, p)
    # ---------------- tier 5: smokes (one clip each) ----------------
    sm_ids = {}
    for nm, kind, arm, tset, _, frames in SMOKES:
        t = T[tset][0]; sm_ids[(arm, tset)] = t["id"]
        if kind == "base": emit(5, "base", arm, tset, [t], [], nm, BASEL[arm][0], BASEL[arm][1])
        elif kind == "hyb": emit(5, "hyb", arm, tset, [t], [], (nm, frames), EST10[arm][res_i(tset)], LOAD10[arm])
        else: emit(5, "ours", arm, tset, [t], [], (nm, frames), ESTL[arm][res_i(tset)], LOADL[arm])
    # ---------------- tier 6: EXP 3 + EXP 2 (10-s matrix), neutral_832 first ----------------
    for tset in SETS10:
        for arm in ("sf24c1hyb", "rt2c1hyb"):
            todo = [x for x in T[tset] if x["id"] != sm_ids.get((arm, tset)) and not os.path.exists("%s/runs/ours_%s_dev_%s/%s/timing.json" % (B, arm, tset, x["id"]))]
            est, load = EST10[arm][res_i(tset)], LOAD10[arm]; per = max(1, int((TARGET10 - load) // est)); k = max(1, math.ceil(len(todo) / per))
            for part in split(todo, k): emit(6, "hyb", arm, tset, part, gates_for(arm, tset), None, est, load)
    # ---------------- tier 6: EXP 1 (P6): baselines first (cheap), then ours per set ----------------
    for model, tset in (("waypoint", "wp_long"), ("mg2", "mg2_long"), ("abot", "abot_long")):
        todo = [x for x in T[tset] if x["id"] != sm_ids.get((model, tset)) and not os.path.exists("%s/runs/%s/%s/timing.json" % (B, model, x["id"]))]
        if todo: emit(6, "base", model, tset, todo, gates_for(model, tset), None, BASEL[model][0], BASEL[model][1])
    for tset in SETS_LONG:
        for arm in ("teacher", "rt2", "adapt25", "sf24", "adapt25c1hyb", "rt2c1hyb", "sf24c1hyb"):
            fz = "%s/runs/ours_%s_%s%s" % (B, arm, "dev_" if arm in HYB else "", tset)
            todo = [x for x in T[tset] if x["id"] != sm_ids.get((arm, tset)) and not os.path.exists("%s/%s/timing.json" % (fz, x["id"]))]
            if not todo: continue
            est, load = ESTL[arm][res_i(tset)], LOADL[arm]; per = max(1, min(CAPL, int((TARGETL - load) // est))); k = max(1, math.ceil(len(todo) / per))
            for part in split(todo, k): emit(6, "hyb" if arm in HYB else "ours", arm, tset, part, gates_for(arm, tset), None, est, load)
    # ---------------- report ----------------
    tot = {}
    for r in plan:
        key = (r["tier"], "EXP 3" if r["arm"] == "sf24c1hyb" and not r["set"].endswith("_long") else ("EXP 2" if r["arm"] == "rt2c1hyb" and not r["set"].endswith("_long") else ("smoke" if r["tier"] == 5 else "EXP 1")))
        s = tot.setdefault(key, dict(chunks=0, clips=0, gpu_h=0.0, pair_h=0.0, single_h=0.0)); s["chunks"] += 1; s["clips"] += r["n"]; h = r["est_s"] / 3600.0
        s["gpu_h"] += h * r["gpus"]; s["pair_h" if r["gpus"] == 2 else "single_h"] += h
        print("tier %d %-40s %-6s %-13s %-17s %3d clips  est %5.0f s (%4.2f h x %d GPU)  gates %s" % (r["tier"], r["name"], r["kind"], r["arm"], r["set"], r["n"], r["est_s"], h, r["gpus"], ",".join(r["gates"]) or "-"))
    for k, s in sorted(tot.items()):
        print("[plan] tier %d %-6s: %3d chunks, %4d clips, est %.1f pair-hours + %.1f single-GPU-hours = %.1f GPU-hours" % (k[0], k[1], s["chunks"], s["clips"], s["pair_h"], s["single_h"], s["gpu_h"]))
    g = sum(s["gpu_h"] for s in tot.values()); print("[plan] total %d chunks, %d clips, %.1f GPU-hours (= %.1f h of the 4 worker GPUs 0-3 if perfectly packed)%s" % (len(plan), sum(r["n"] for r in plan), g, g / 4, " (DRY RUN)" if a.dry_run else ""))
    if not a.dry_run:
        with open(Q + "/INDEX.tsv", "a") as f:
            f.write("# tiers 5-6 [10/19 EXP 1/2/3], %s: gen/make_queue_long.py wrote %d chunks (%d clips, est %.1f GPU-hours): tier 5 = smokes -> gen/sentinels/<smoke>.ok|.fail; tier 6 = gated runs (see each script's gates=)\n" % (MADE, len(plan), sum(r["n"] for r in plan), g))
        json.dump(plan, open(Q + "/plan_long_1019.json", "w"), indent=1)
    return 0
if __name__ == "__main__": sys.exit(main())
