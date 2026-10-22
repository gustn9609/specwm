#!/usr/bin/env python3
"""[10/19 audit] Queue tier 9: the RE-CALIBRATED adaptive arms  adapt25r  (= adapt25 with two_machine --adapt-thr <baselines/adapt_thr.json by_set[SET].thr>)
and  adapt25rc1hyb  (= adapt25c1hyb with the same device flag; cloud unchanged) on the 10-s sets mg2, wp, abot, astra, neutral_832 (run_ours.sh adapt25r |
adapt25rc1hyb; thresholds = the 75th percentile of the step-1 head score over chunks >= 3 of a DISJOINT scene set at the same resolution, see adapt_thr.json
"rule"; the adapt25 arm's result.json threshold 0.2869 extended only 1-10 % of the chunks on the frozen sets).  Runs automatically by the gen/worker.sh
workers once tiers <= 8 have nothing left to claim (tier dirs in numeric order; .sh chunks as `bash FILE G`).  GPUs 0-3 only (pairs 0+1 / 2+3).
Tier 9 = first ONE smoke chunk per arm (NNN_smoke_adapt25r_mg2.sh / NNN_smoke_adapt25rc1hyb_mg2.sh: the first FROZEN mg2 clip, 1 clip each) ->
gen/smoke_check.py (--kind ours / hyb, 161 frames) + a threshold check (the RAW timing.json of the clip has adapt_thr_source "--adapt-thr ..." and adapt_thr ==
adapt_thr.json's value for the set) -> gen/sentinels/adapt25r_mg2.ok|.fail / adapt25rc1hyb_mg2.ok|.fail; then the full chunks of each set, each FIRST gating
on its arm's sentinel exactly as tier 6/7 (gate(): .ok -> go; .fail -> exit 1; neither and the smoke chunk neither pending in queue/9 nor running in
queue/claimed -> exit 1; else wait <= GATE_WAIT s (default 3600) WITHOUT holding a GPU lock).  Every tier-9 script also waits (need_arm, same GATE_WAIT, no
lock) until run_ours.sh has a `case` entry for its arm: the chunks were queued BEFORE the arm entries were added to run_ours.sh (another agent edits it).
Chunk sizing = tier 6/7's rule: load + n x clip <= TARGET 3000 s (per-clip / load estimates below), balanced split; numbered from --start (default 320).
Conventions: make_queue_ours.load_tasks (FROZEN order), OURS_CHUNK / HYB_CHUNK header lines, chunkdef json heredoc, the tier-6 pair mechanics
(flock gpu<lo> then gpu<hi>; cloud on lo, device on hi), skip-if-done through run_ours.sh.
usage: python3 /home/hk28965/specwm/baselines/gen/make_queue_adapt25r.py [--sets mg2,wp,abot,astra,neutral_832] [--smoke-set mg2] [--start 320] [--dry-run]
(plain python3; re-running adds only clips not yet queued / frozen)"""
import os, sys, json, math, time, glob, argparse
B = "/home/hk28965/specwm/baselines"; GEN = B + "/gen"; sys.path.insert(0, GEN)
import make_queue_ours as MQ          # load_tasks (FROZEN order)
TIER = 9; SOLO = "adapt25r"; HYB = "adapt25rc1hyb"
ap = argparse.ArgumentParser(); ap.add_argument("--sets", default="mg2,wp,abot,astra,neutral_832"); ap.add_argument("--smoke-set", default="mg2", help="one smoke clip per arm (the set's first FROZEN clip)")
ap.add_argument("--start", type=int, default=320, help="first NNN (never below 1 + the highest NNN in use)"); ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--queue-root", default=GEN + "/queue"); a = ap.parse_args()
Q = a.queue_root; MADE = time.strftime("%Y-%m-%d %H:%M:%S"); RES = ["1024x512", "640x352", "1280x704", "832x480"]
# per-clip seconds inside a warm hold by resolution 1024x512 / 640x352 / 1280x704 / 832x480 (10/19 audit estimates from the frozen adapt25 / adapt25c1hyb holds:
# solo 26 / 8.8 / 47 / 17.5 (astra = neutral_832 = 17.5), hybrid pair 42 / 14 / 80 / 30 (astra 23)) and load s per chunk process (solo ~300, pair ~520 serial)
EST = {SOLO: (26, 8.8, 47, 17.5), HYB: (42, 14, 80, 30)}; EST_SET = {(SOLO, "astra"): 17.5, (HYB, "astra"): 23}; LOAD = {SOLO: 300, HYB: 520}; TARGET = 3000
GATE_SH = r'''SENT=$B/gen/sentinels; QR=$B/gen/queue; GATE_WAIT=${GATE_WAIT:-3600}; mkdir -p $SENT
gate() {   # $1 = smoke name: go on when $SENT/$1.ok exists; exit 1 at once on $SENT/$1.fail or when the smoke chunk is neither pending (queue/9) nor running; else wait (no lock held)
  local n=$1 t0=$(date +%s)
  while :; do
    [ -f $SENT/$n.ok ] && { echo "[gate] smoke $n ok"; return 0; }
    [ -f $SENT/$n.fail ] && { echo "[gate] smoke $n FAILED: $(head -c 400 $SENT/$n.fail | tr '\n' ' ') -> exit 1"; exit 1; }
    ls $QR/9/*_smoke_$n.sh $QR/claimed/*_smoke_$n.sh.gpu* > /dev/null 2>&1 || { echo "[gate] smoke $n: no sentinel and its chunk is neither pending nor running (failed / killed?) -> exit 1"; exit 1; }
    [ $(( $(date +%s) - t0 )) -ge $GATE_WAIT ] && { echo "[gate] smoke $n still pending/running after ${GATE_WAIT} s -> exit 1 (re-queue this chunk once the smoke is done)"; exit 1; }
    sleep 30
  done
}
need_arm() {   # [10/19 audit] $1 = arm: wait (no lock held) until run_ours.sh has a `case` entry for it (the tier-9 chunks were queued before the arm entries were added)
  local t0=$(date +%s)
  while ! grep -qE "^[[:space:]]*([A-Za-z0-9_]+\|)*$1(\|[A-Za-z0-9_]+)*\)" $B/run_ours.sh; do
    [ $(( $(date +%s) - t0 )) -ge $GATE_WAIT ] && { echo "[gate] arm $1 still has no case entry in run_ours.sh after ${GATE_WAIT} s -> exit 1 (add it, re-queue this chunk)"; exit 1; }
    sleep 30
  done; echo "[gate] arm $1 present in run_ours.sh"
}'''
# after smoke_check.py: the clip really ran with the per-set --adapt-thr (RAW timing.json kept by --purge; @RAW@ = the solo task dir or the hybrid device task dir)
THR_CHECK = r'''
if [ $RC -eq 0 ]; then   # [10/19 audit] threshold check on top of smoke_check.py: adapt_thr_source "--adapt-thr ..." and adapt_thr == adapt_thr.json by_set[@SET@].thr
  python3 - "@RAW@" "@IDS@" "$SENT/@SMOKE@" "@SET@" <<'EOF_THR'; RC=$?
import json, os, sys
raw, ids, sent, tset = sys.argv[1], sys.argv[2].split(","), sys.argv[3], sys.argv[4]; bad = []; got = {}
want = json.load(open("/home/hk28965/specwm/baselines/adapt_thr.json"))["by_set"][tset]["thr"]
for i in ids:
    p = os.path.join(raw, i, "timing.json")
    if not os.path.exists(p): bad.append("%s: no raw timing.json at %s" % (i, p)); continue
    T = json.load(open(p)); got[i] = dict(adapt_thr=T.get("adapt_thr"), adapt_thr_source=T.get("adapt_thr_source"), n_ext=T.get("n_ext"), n_chunks_adapt=len(T.get("adapt") or []))
    if not str(T.get("adapt_thr_source", "")).startswith("--adapt-thr"): bad.append("%s: adapt_thr_source %r (the arm did not pass --adapt-thr)" % (i, T.get("adapt_thr_source")))
    if T.get("adapt_thr") is None or abs(float(T["adapt_thr"]) - want) > 1e-6: bad.append("%s: adapt_thr %s != adapt_thr.json %s" % (i, T.get("adapt_thr"), want))
    if not T.get("adapt"): bad.append("%s: no 'adapt' log (head scores) in the raw timing.json" % i)
ok = sent + ".ok"; fail = sent + ".fail"
if bad:
    d = json.load(open(ok)) if os.path.exists(ok) else {}; d.update(ok=False, thr_problems=bad, thr=got)
    json.dump(d, open(fail, "w"), indent=1); os.path.exists(ok) and os.remove(ok); print("[thr] FAIL", bad); sys.exit(1)
d = json.load(open(ok)); d["thr"] = dict(want=want, clips=got); json.dump(d, open(ok + ".tmp", "w"), indent=1); os.replace(ok + ".tmp", ok); print("[thr] ok", want, got); sys.exit(0)
EOF_THR
fi'''
def thr_check(d, smoke, raw):
    return THR_CHECK.replace("@RAW@", raw).replace("@IDS@", ",".join(d["ids"])).replace("@SMOKE@", smoke).replace("@SET@", d["set"])
def ours_script(name, d, gates, smoke=None):
    """single-GPU chunk (= make_queue_long.ours_script, tier 9, GPUs 0-3, + need_arm; OURS_CHUNK line kept for the queued-ids scans)"""
    stem = name[:-3]; js = json.dumps(d, indent=1); assert "\nEOF_CHUNK\n" not in js; arm = d["arm"]
    chk = ("\npython3 $B/gen/smoke_check.py %s --kind ours --runs $B/runs/ours_%s_%s --ids %s --frames 161 --rc $RC --note \"$NAME hold ${HELD:-?} s peak ${PEAK:-?} MiB\"; RC=$?"
           % (smoke, arm, d["set"], ",".join(d["ids"])) + thr_check(d, smoke, "$B/raw/ours_%s_%s" % (arm, d["set"]))) if smoke else ""
    return """#!/bin/bash
# generated by gen/make_queue_adapt25r.py %(made)s -- tier %(tier)d: OUR %(arm)s (re-calibrated adapt threshold, baselines/adapt_thr.json) on %(set)s, %(n)d clips at %(res)s, ONE GPU%(smk)s (re-run make_queue_adapt25r.py, do not edit)
# OURS_CHUNK set=%(set)s arm=%(arm)s n=%(n)d gates=%(gates)s ids=%(ids)s
# usage: bash THIS GPU   (worker.sh runs .sh chunks so, WITHOUT a lock): wait for the arm's case entry in run_ours.sh, gate on the tier-9 smoke sentinel (none for a smoke itself), then
# flock /tmp/gpu$G.lock env CUDA_VISIBLE_DEVICES=$G _gpu_peak_run.sh run_ours.sh %(set)s %(arm)s $G CHUNKJSON (skip-if-done on runs/ours_%(arm)s_%(set)s).
G=${1:?usage: $0 GPU}; B=/home/hk28965/specwm/baselines; NAME=%(stem)s
case "$G" in 0|1|2|3) ;; *) echo "[queue] $NAME: GPU $G: tier 9 runs only on GPUs 0-3 (10/19: GPUs 4-7 not ours)"; exit 1;; esac
%(gate_sh)s
need_arm %(arm)s
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
""" % dict(made=MADE, tier=TIER, arm=arm, set=d["set"], n=d["n"], res=d["res"], smk=" [SMOKE %s]" % smoke if smoke else "", gates=",".join(gates) or "-",
           ids=",".join(d["ids"]), stem=stem, gate_sh=GATE_SH, gate_calls="\n".join("gate %s" % g for g in gates), js=js, chk=chk)
def hyb_script(name, d, gates, smoke=None):
    """GPU-pair chunk (= make_queue_long.hyb_script, tier 9, pairs 0+1 / 2+3, + need_arm; HYB_CHUNK line kept for make_queue_ours.queued_ids)"""
    stem = name[:-3]; js = json.dumps(d, indent=1); assert "\nEOF_CHUNK\n" not in js; arm = d["arm"]
    chk = ("\npython3 $B/gen/smoke_check.py %s --kind hyb --runs $B/runs/ours_%s_dev_%s --runs-cloud $B/runs/ours_%s_cloud_%s --ids %s --frames 161 --rc $RC --note \"$NAME (see HYB_HOLD in logs/ours_%s_%s.log)\"; RC=$?"
           % (smoke, arm, d["set"], arm, d["set"], ",".join(d["ids"]), arm, d["set"]) + thr_check(d, smoke, "$B/raw/ours_%s_%s/device" % (arm, d["set"]))) if smoke else ""
    return """#!/bin/bash
# generated by gen/make_queue_adapt25r.py %(made)s -- tier %(tier)d: OUR hybrid %(arm)s (adapt25c1hyb with the re-calibrated adapt threshold, baselines/adapt_thr.json; cloud unchanged) on %(set)s, %(n)d clips at %(res)s, on a GPU PAIR%(smk)s (re-run make_queue_adapt25r.py, do not edit)
# HYB_CHUNK set=%(set)s arm=%(arm)s n=%(n)d gates=%(gates)s ids=%(ids)s
# usage: bash THIS GPU   (worker.sh runs .sh chunks so, WITHOUT a lock): wait for the arm's case entry in run_ours.sh, gate on the tier-9 smoke sentinel first (no lock held while waiting), then the tier-6
# pair mechanics: pair of G = G xor 1 (ONLY 0+1 and 2+3: GPUs 4-7 are not ours; G 4-7 exits 1), cloud on the lower index, device on the higher, port 32300 + lower; flock /tmp/gpu<lo>.lock THEN
# /tmp/gpu<hi>.lock, run_ours.sh %(set)s %(arm)s G CHUNKJSON (it sees both locks held by its ancestors). Idempotent (runs/ours_%(arm)s_dev_%(set)s).
G=${1:?usage: $0 GPU}; B=/home/hk28965/specwm/baselines; NAME=%(stem)s
case "$G" in 0|1|2|3) ;; *) echo "[queue] $NAME: GPU $G: tier-9 hybrids run only on the pairs 0+1 and 2+3 (GPUs 4-7 not ours)"; exit 1;; esac
PG=$((G ^ 1)); LO=$(( G < PG ? G : PG )); HI=$(( G < PG ? PG : G ))
%(gate_sh)s
need_arm %(arm)s
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
""" % dict(made=MADE, tier=TIER, arm=arm, set=d["set"], n=d["n"], res=d["res"], smk=" [SMOKE %s]" % smoke if smoke else "", gates=",".join(gates) or "-",
           ids=",".join(d["ids"]), stem=stem, gate_sh=GATE_SH, gate_calls="\n".join("gate %s" % g for g in gates), js=js, chk=chk)
def split(lst, k):
    n = len(lst); return [lst[i * n // k:(i + 1) * n // k] for i in range(k)] if n else []
def main():
    sets = [s for s in a.sets.split(",") if s]; assert a.smoke_set in sets
    for sub in (str(TIER), "claimed", "done", "failed"): os.makedirs(os.path.join(Q, sub), exist_ok=True)
    os.makedirs(GEN + "/sentinels", exist_ok=True)
    queued = set()   # (arm, set, id) already inside a tier-9 chunk of this maker (pending / claimed / done)
    for f in glob.glob("%s/%d/*.sh" % (Q, TIER)) + glob.glob("%s/claimed/*.sh.gpu*" % Q) + glob.glob("%s/done/*.sh.gpu*" % Q):
        try:
            head = open(f).read(800)
            if "generated by gen/make_queue_adapt25r.py" not in head: continue
            for ln in head.splitlines():
                if ln.startswith(("# HYB_CHUNK ", "# OURS_CHUNK ")):
                    kv = dict(x.split("=", 1) for x in ln.split()[2:] if "=" in x); queued.update((kv.get("arm"), kv.get("set"), i) for i in kv.get("ids", "").split(",") if i)
                    break
        except Exception: continue
    used = max([int(os.path.basename(f).split("_")[0]) for sub in [d for d in os.listdir(Q) if d.isdigit()] + ["claimed", "done", "failed"]
                for f in glob.glob("%s/%s/*" % (Q, sub)) if os.path.basename(f).split("_")[0].isdigit()] or [-1])
    idx = max(a.start, used + 1)   # after every NNN in use (all tiers), never below --start
    plan = []; T = {}
    for s in sets:
        t, _ = MQ.load_tasks(s, False); assert t, s; T[s] = t
    def est_of(arm, tset): return EST_SET.get((arm, tset), EST[arm][RES.index(T[tset][0]["res"])])
    def emit(arm, tset, tasks, gates, smoke=None):
        nonlocal idx
        tasks = [x for x in tasks if (arm, tset, x["id"]) not in queued]
        if not tasks: return
        name = "%03d_%s.sh" % (idx, ("smoke_%s" % smoke) if smoke else "%s_%s" % (tset, arm)); idx += 1
        d = dict(set=tset, arm=arm, tier=TIER, res=tasks[0]["res"], area=tasks[0]["area"], n=len(tasks), ids=[x["id"] for x in tasks], made=MADE, tasks=tasks, gates=gates)
        sec = LOAD[arm] + len(tasks) * est_of(arm, tset); gpus = 2 if arm == HYB else 1
        plan.append(dict(tier=TIER, name=name, kind="hyb" if arm == HYB else "ours", arm=arm, set=tset, n=len(tasks), gpus=gpus, est_s=sec, gates=gates, smoke=smoke))
        if not a.dry_run:
            p = "%s/%d/%s" % (Q, TIER, name); tmp = p + ".tmp"; open(tmp, "w").write((hyb_script if arm == HYB else ours_script)(name, d, gates, smoke)); os.chmod(tmp, 0o755); os.replace(tmp, p)
    sm = {arm: "%s_%s" % (arm, a.smoke_set) for arm in (SOLO, HYB)}; sm_id = T[a.smoke_set][0]["id"]
    for arm in (SOLO, HYB): emit(arm, a.smoke_set, [T[a.smoke_set][0]], [], smoke=sm[arm])
    for tset in sets:
        for arm in (SOLO, HYB):
            fz = "%s/runs/ours_%s_%s%s" % (B, arm, "dev_" if arm == HYB else "", tset)
            todo = [x for x in T[tset] if not (tset == a.smoke_set and x["id"] == sm_id) and not os.path.exists("%s/%s/timing.json" % (fz, x["id"]))]
            if not todo: continue
            est = est_of(arm, tset); per = max(1, int((TARGET - LOAD[arm]) // est)); k = max(1, math.ceil(len(todo) / per))
            for part in split(todo, k): emit(arm, tset, part, [sm[arm]])
    gh = ph = sh = 0.0
    for r in plan:
        h = r["est_s"] / 3600.0; gh += h * r["gpus"]; ph += h if r["gpus"] == 2 else 0; sh += h if r["gpus"] == 1 else 0
        print("tier %d %-34s %-5s %-14s %-12s %3d clips  est %5.0f s (%4.2f h x %d GPU)  gates %s" % (r["tier"], r["name"], r["kind"], r["arm"], r["set"], r["n"], r["est_s"], h, r["gpus"], ",".join(r["gates"]) or "-"))
    print("[plan] tier %d: %d chunks, %d clips, est %.1f pair-hours + %.1f single-GPU-hours = %.1f GPU-hours%s" % (TIER, len(plan), sum(r["n"] for r in plan), ph, sh, gh, " (DRY RUN)" if a.dry_run else ""))
    if not a.dry_run and plan:
        with open(Q + "/INDEX.tsv", "a") as f:
            f.write("# tier 9 [10/19 audit: re-calibrated adapt threshold, baselines/adapt_thr.json], %s: gen/make_queue_adapt25r.py wrote %d chunks (%d clips, est %.1f pair-hours + %.1f single-GPU-hours = %.1f GPU-hours) on sets %s: arms adapt25r (solo) + adapt25rc1hyb (pair); smokes %s -> gen/sentinels/adapt25r_%s.ok|.fail / adapt25rc1hyb_%s.ok|.fail (smoke_check + threshold check); every chunk waits for its arm's case entry in run_ours.sh (need_arm) and gates on its arm's smoke; runs/ours_adapt25r_<set>, runs/ours_adapt25rc1hyb_{dev,cloud}_<set>\n"
                    % (MADE, len(plan), sum(r["n"] for r in plan), ph, sh, gh, ",".join(sets), ",".join(r["name"] for r in plan if r["smoke"]), a.smoke_set, a.smoke_set))
        json.dump(plan, open(Q + "/plan_adapt25r_1019.json", "w"), indent=1)
    return 0
if __name__ == "__main__": sys.exit(main())
