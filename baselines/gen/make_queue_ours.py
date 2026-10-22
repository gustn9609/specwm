#!/usr/bin/env python3
"""[10/17 B frozen baselines] Chunk driver for OUR arms: split every (set, arm) of the frozen matrix into chunk jsons sized for a <= 20-min GPU lock hold
and drop them into the worker queue  baselines/gen/queue/<tier>/<NNN>_<set>_<arm>.json  (consumed by baselines/gen/worker.sh GPU, which runs
flock /tmp/gpu<G>.lock env CUDA_VISIBLE_DEVICES=<G> baselines/run_ours.sh SET ARM G CHUNK).  Plain python3 (no numpy).
NB file name: the task asked for gen/make_queue.py, but that path is taken by the BASELINE chunk maker (tier-1 *.sh job files, written concurrently by
the baseline agent), so OUR chunk maker lives here as make_queue_ours.py; both feed the same queue dirs and the same worker.

Tiers (priority order of the worker):  1 = teacher + rt2 on all sets;  2 = adapt25 + sf24 on all sets;
                                       3 = student1p3b on all sets + sf1p3b_cal / sf1p3b_pub on astra, neutral_832, neutral_832_nocap (832x480 only)
Sets = wp mg2 abot astra neutral_wp neutral_mg2 neutral_abot neutral_832 neutral_832_nocap  (baselines/tasks/<set>.json from make_tasks.py; a set whose
json is missing is reported and skipped, re-run once it exists).
Frozen scene subsets (orchestrator decision): the matrix uses only the scenes listed in baselines/scenes/<scene set>/FROZEN.json (scene set = wp | mg2 |
abot | astra | neutral; neutral_* task sets share scenes/neutral).  When a FROZEN.json is missing it is written here (O_EXCL, never overwritten) by the
rule: wp = the 12 first sorted Biome seeds (not the starter_* images), mg2 = 8 first sorted universal_* + 4 first sorted mc_*, abot = the 7 web_client
presets (non-case* names) + the 5 first case* posters, astra = all 3, neutral = all 12.  --all-scenes ignores FROZEN.json (the baselines may run their
remaining scenes later; the frozen matrix is the subset).
Chunk sizes (clips per chunk, gap 6; compile warm-up included) by arm x resolution 1024x512 / 640x352 / 1280x704 / 832x480:
  teacher 11/22/6/14   rt2 15/31/8/20   adapt25 14/29/8/19   sf24 13/27/7/17   student1p3b 16/32/9/21   sf1p3b_cal, sf1p3b_pub 22 (832x480 only)
Skips: a task whose frozen result baselines/runs/ours_<arm>_<set>/<id>/timing.json exists, and a task already inside a pending or claimed chunk of the same
(set, arm) (re-running is idempotent; a chunk in queue/failed/ is NOT counted, so its unfinished tasks are re-queued).  NNN continues after the highest
index already used in that tier dir and in claimed/done/failed (the baseline .sh chunks count too), so names never collide and new chunks run after the
ones already queued.
Chunk json = {"set", "arm", "tier", "res", "area", "n", "ids", "made", "tasks": [the task entries, two_machine schema]} — self-contained: run_ours.sh
reads the "tasks" list directly (it writes a plain-list copy for the loaders).  Task order inside a (set, arm): FROZEN scene order, programs
P1 P2 P2p P3 P4 P4p P5.
Tier 4 [10/17 C] = OUR headline hybrid adapt25c1hyb (device adapt25 + cloud 1 step at t=750 + KV feedback) on all sets, on a GPU PAIR (0+1, 2+3, 4+5):
  NOT a .json chunk but a self-contained .sh chunk  queue/4/NNN_<set>_adapt25c1hyb.sh  (worker.sh runs .sh chunks as `bash FILE G` without a lock): it
  writes its chunk json (heredoc) to raw/ours_adapt25c1hyb_<set>/chunkdef_<name>.json and runs
  flock /tmp/gpu<lo>.lock flock /tmp/gpu<hi>.lock run_ours.sh SET adapt25c1hyb G CHUNKJSON  (ascending lock order: no dead-lock between the two workers of a
  pair; run_ours.sh sees both locks held by its ancestors and does not re-take them). Its ids are on a "# HYB_CHUNK set=.. arm=.. ids=a,b,c" line (read by
  queued_ids). Skip-if-done on runs/ours_adapt25c1hyb_dev_<set>. --tiers 4 must be asked for explicitly (default stays 1,2,3).
  Sizes adapt25c1hyb (hold <= ~25 min incl. ~9-10 min serial load of the two 14B processes; HYB_CLIP_S / HYB_LOAD_S): 11 / 27 / 6 / 15 clips at 1024x512 / 640x352 / 1280x704 / 832x480.
--ids a,b,c restricts every (set, arm) to these task ids (e.g. a one-clip smoke chunk with --queue-root in a scratch dir).
usage: python3 /home/hk28965/specwm/baselines/gen/make_queue_ours.py [--tiers 1,2,3] [--sets a,b] [--arms x,y] [--dry-run] [--all-scenes]
       [--queue-root DIR] [--frozen-only]   ;  --frozen-only just writes the missing FROZEN.json files and prints them."""
import os, sys, json, glob, time, argparse
B = "/home/hk28965/specwm/baselines"
TIERS = {1: ["teacher", "rt2"], 2: ["adapt25", "sf24"], 3: ["student1p3b", "sf1p3b_cal", "sf1p3b_pub"], 4: ["adapt25c1hyb"]}
HYB = "adapt25c1hyb"
def frozen_dir(runs, arm, tset): return "%s/ours_%s_dev_%s" % (runs, arm, tset) if arm == HYB else "%s/ours_%s_%s" % (runs, arm, tset)
ALL_SETS = ["wp", "mg2", "abot", "astra", "neutral_wp", "neutral_mg2", "neutral_abot", "neutral_832", "neutral_832_nocap"]
ARM_SETS = {"sf1p3b_cal": ["astra", "neutral_832", "neutral_832_nocap"], "sf1p3b_pub": ["astra", "neutral_832", "neutral_832_nocap"]}   # default: ALL_SETS
RES = ["1024x512", "640x352", "1280x704", "832x480"]
# [10/17 C] hybrid per-clip wall (s, one clip inside a warm hold: device + cloud prepare, 14 chunks, both full decodes, the conversion of both streams)
# by resolution 1024x512 / 640x352 / 1280x704 / 832x480 -> clips per chunk = floor((1500 - HYB_LOAD_S) / clip_s)   (hold target 25 min)
# measured 10/17 smoke (neutral_832__easy_lake__P2, GPUs 4+5): load is SERIAL (device starts once the cloud listens): cloud load 252 s + device load/warm 288 s,
# then 49 s for the clip (gen 24.2 s) + 10 s conversion of both streams; hold 595 s for 1 clip. 832 clip ~58 s; other resolutions = 0.78 x the teacher's
# per-clip wall in the frozen holds (teacher 101 / 41 / 176 / 75 s), 1280x704 rounded up (window 12, untested in hybrid mode).
HYB_CLIP_S = (80, 33, 150, 58); HYB_LOAD_S = 580
SIZES = {"teacher": (11, 22, 6, 14), "rt2": (15, 31, 8, 20), "adapt25": (14, 29, 8, 19), "sf24": (13, 27, 7, 17), "student1p3b": (16, 32, 9, 21),
         "sf1p3b_cal": (None, None, None, 22), "sf1p3b_pub": (None, None, None, 22)}
PROG_ORDER = ["P1", "P2", "P2p", "P3", "P4", "P4p", "P5"]
def scene_set(task_set): return "neutral" if task_set.startswith("neutral") else task_set
def frozen_rule(sset, names):
    names = sorted(names)
    if sset == "wp": return [n for n in names if not n.startswith("starter_")][:12]
    if sset == "mg2": return [n for n in names if n.startswith("universal_")][:8] + [n for n in names if n.startswith("mc_")][:4]
    if sset == "abot": return [n for n in names if not n.startswith("case")] + [n for n in names if n.startswith("case")][:5]
    return names   # astra (3), neutral (12)
def frozen_scenes(sset, write=True):
    """FROZEN.json of a scene set (list of scene names); written by the rule when missing (O_EXCL: a concurrent writer wins, we re-read)"""
    d = "%s/scenes/%s" % (B, sset); p = d + "/FROZEN.json"
    if os.path.exists(p): return json.load(open(p)), False
    names = sorted({os.path.splitext(os.path.basename(f))[0] for f in glob.glob(d + "/*.png") + glob.glob(d + "/*.jpg")})
    assert names, "no scene images in " + d
    fr = frozen_rule(sset, names)
    if not write: return fr, False
    try:
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "w") as f: json.dump(fr, f, indent=1); f.write("\n")
        return fr, True
    except FileExistsError: return json.load(open(p)), False
def prog_key(p): return PROG_ORDER.index(p) if p in PROG_ORDER else 100 + len(p)
def load_tasks(tset, all_scenes):
    p = "%s/tasks/%s.json" % (B, tset)
    if not os.path.exists(p): return None, None
    t = json.load(open(p)); t = t["tasks"] if isinstance(t, dict) else t
    fr, _ = frozen_scenes(scene_set(tset)) if not all_scenes else (sorted({x["scene"] for x in t}), False)
    order = {s: i for i, s in enumerate(fr)}
    t = [x for x in t if x["scene"] in order]
    t.sort(key=lambda x: (order[x["scene"]], prog_key(x["program"]), x["id"]))
    return t, fr
def queued_ids(q, tier, tset, arm):
    """task ids inside pending (queue/<tier>/*.json) and claimed (queue/claimed/*.json.gpu*) chunks of this (set, arm); [10/17 C] plus the tier-4 .sh chunks
    (pending queue/<tier>/*.sh, claimed/*.sh.gpu*) via their "# HYB_CHUNK set=.. arm=.. ids=.." line"""
    ids = set()
    for f in glob.glob("%s/%d/*.sh" % (q, tier)) + glob.glob("%s/claimed/*.sh.gpu*" % q):
        try:
            for ln in open(f):
                if ln.startswith("# HYB_CHUNK "):
                    kv = dict(x.split("=", 1) for x in ln.split()[2:] if "=" in x)
                    if kv.get("set") == tset and kv.get("arm") == arm: ids.update(i for i in kv.get("ids", "").split(",") if i)
                    break
        except Exception: continue
    for f in glob.glob("%s/%d/*.json" % (q, tier)) + glob.glob("%s/claimed/*.json.gpu*" % q) + glob.glob("%s/claimed/*.json" % q):
        try: d = json.load(open(f))
        except Exception: continue
        if isinstance(d, dict) and d.get("set") == tset and d.get("arm") == arm: ids.update(d.get("ids") or [x["id"] for x in d.get("tasks", [])])
    return ids
def next_index(q, tier):
    """1 + the highest leading NNN among this tier's pending chunks (.json and the baselines' .sh) and everything in claimed/done/failed"""
    n = -1
    for sub in ("%d" % tier, "claimed", "done", "failed"):
        for f in glob.glob("%s/%s/*" % (q, sub)):
            head = os.path.basename(f).split("_")[0]
            if head.isdigit(): n = max(n, int(head))
    return n + 1
def hyb_script(name, d, tier):
    """[10/17 C] self-contained tier-4 chunk script for the hybrid arm (GPU pair, both locks ascending, then run_ours.sh under them)"""
    stem = name[:-3]; js = json.dumps(d, indent=1); assert "\nEOF_CHUNK\n" not in js
    return """#!/bin/bash
# generated by gen/make_queue_ours.py %(made)s -- tier %(tier)d: OUR hybrid adapt25c1hyb on %(set)s, %(n)d clips at %(res)s, on a GPU PAIR (re-run make_queue_ours.py, do not edit)
# HYB_CHUNK set=%(set)s arm=adapt25c1hyb n=%(n)d ids=%(ids)s
# usage: bash THIS GPU   (worker.sh runs .sh chunks so, WITHOUT a lock). Pair of GPU G = G xor 1 (0+1, 2+3, 4+5): cloud on the lower index, device on the
# higher, port 32300 + lower. Takes flock /tmp/gpu<lo>.lock THEN /tmp/gpu<hi>.lock (ascending: two hybrid chunks claimed by both workers of a pair just
# queue up) and runs run_ours.sh SET adapt25c1hyb G CHUNKJSON under both (it sees the locks held by its ancestors and does not re-take them).
# Idempotent: tasks frozen in runs/ours_adapt25c1hyb_dev_%(set)s are skipped. Hold time + nvidia-smi peak of both GPUs: HYB_HOLD in logs/ours_adapt25c1hyb_%(set)s.log.
G=${1:?usage: $0 GPU}; B=/home/hk28965/specwm/baselines; NAME=%(stem)s
case "$G" in 0|1|2|3|4|5) ;; *) echo "[queue] $NAME: GPU $G has no pair (0+1, 2+3, 4+5)"; exit 1;; esac
PG=$((G ^ 1)); LO=$(( G < PG ? G : PG )); HI=$(( G < PG ? PG : G ))
CJ=$B/raw/ours_adapt25c1hyb_%(set)s/chunkdef_$NAME.json; mkdir -p $B/raw/ours_adapt25c1hyb_%(set)s
cat > $CJ.tmp <<'EOF_CHUNK'
%(js)s
EOF_CHUNK
mv -f $CJ.tmp $CJ
t0=$(date +%%s); echo "[queue $(date '+%%m/%%d %%H:%%M:%%S')] $NAME: %(n)d clips, waiting for /tmp/gpu$LO.lock then /tmp/gpu$HI.lock (claimed by gpu $G)"
flock /tmp/gpu$LO.lock flock /tmp/gpu$HI.lock env CUDA_VISIBLE_DEVICES=$LO,$HI $B/run_ours.sh %(set)s adapt25c1hyb $G $CJ; RC=$?
echo "[queue $(date '+%%m/%%d %%H:%%M:%%S')] $NAME: rc $RC, wall $(( $(date +%%s) - t0 )) s (lock wait + hold; hold in HYB_HOLD)"
exit $RC
""" % dict(made=d["made"], tier=tier, set=d["set"], n=d["n"], res=d["res"], ids=",".join(d["ids"]), stem=stem, js=js)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tiers", default="1,2,3"); ap.add_argument("--sets", default=""); ap.add_argument("--arms", default="")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--all-scenes", action="store_true", help="ignore FROZEN.json (every scene of the tasks json)")
    ap.add_argument("--queue-root", default=B + "/gen/queue"); ap.add_argument("--frozen-only", action="store_true", help="only write the missing scenes/<set>/FROZEN.json files")
    ap.add_argument("--runs", default=B + "/runs"); ap.add_argument("--ids", default="", help="[10/17 C] only these task ids (comma list)"); a = ap.parse_args()
    ids_f = set(a.ids.split(",")) if a.ids else None
    for sset in ("wp", "mg2", "abot", "astra", "neutral"):
        fr, new = frozen_scenes(sset); print("[frozen] %-7s %2d scenes %s: %s" % (sset, len(fr), "WRITTEN" if new else "(exists)", " ".join(fr)))
    if a.frozen_only: return 0
    q = a.queue_root; tiers = [int(x) for x in a.tiers.split(",") if x]; sets_f = set(a.sets.split(",")) if a.sets else None; arms_f = set(a.arms.split(",")) if a.arms else None
    for sub in ["claimed", "done", "failed"] + ["%d" % t for t in tiers]: os.makedirs(os.path.join(q, sub), exist_ok=True)
    made = []; tot = dict(tasks=0, done=0, queued=0, new=0, chunks=0)
    for tier in tiers:
        idx = next_index(q, tier)
        for arm in TIERS[tier]:
            if arms_f and arm not in arms_f: continue
            for tset in ARM_SETS.get(arm, ALL_SETS):
                if sets_f and tset not in sets_f: continue
                tasks, fr = load_tasks(tset, a.all_scenes)
                if tasks and ids_f: tasks = [x for x in tasks if x["id"] in ids_f]
                if tasks is None: print("[skip] tier %d %-18s %-12s no tasks json baselines/tasks/%s.json (make_tasks.py)" % (tier, tset, arm, tset)); continue
                if not tasks: print("[skip] tier %d %-18s %-12s no task of a frozen scene" % (tier, tset, arm)); continue
                res = {x["res"] for x in tasks}; assert len(res) == 1, (tset, res); res = res.pop(); area = {x["area"] for x in tasks}.pop()
                size = SIZES[arm][RES.index(res)] if arm != HYB else (max(1, int((1500 - HYB_LOAD_S) // HYB_CLIP_S[RES.index(res)])) if HYB_CLIP_S[RES.index(res)] else None)
                if size is None: print("[skip] tier %d %-18s %-12s arm not defined at %s" % (tier, tset, arm, res)); continue
                rd = frozen_dir(a.runs, arm, tset); done = [x for x in tasks if os.path.exists("%s/%s/timing.json" % (rd, x["id"]))]
                qd = queued_ids(q, tier, tset, arm); todo = [x for x in tasks if x["id"] not in qd and not os.path.exists("%s/%s/timing.json" % (rd, x["id"]))]
                nq = len([x for x in tasks if x["id"] in qd]); chunks = [todo[i:i + size] for i in range(0, len(todo), size)]
                tot["tasks"] += len(tasks); tot["done"] += len(done); tot["queued"] += nq; tot["new"] += len(todo); tot["chunks"] += len(chunks)
                print("[plan] tier %d %-18s %-12s %s: %3d tasks (%d scenes x %d programs), %3d done, %3d already queued, %3d new -> %2d chunks of <= %d" %
                      (tier, tset, arm, res, len(tasks), len({x["scene"] for x in tasks}), len({x["program"] for x in tasks}), len(done), nq, len(todo), len(chunks), size))
                for ch in chunks:
                    name = "%03d_%s_%s.%s" % (idx, tset, arm, "sh" if arm == HYB else "json"); p = "%s/%d/%s" % (q, tier, name); idx += 1
                    d = dict(set=tset, arm=arm, tier=tier, res=res, area=area, n=len(ch), ids=[x["id"] for x in ch], made=time.strftime("%Y-%m-%d %H:%M:%S"), tasks=ch)
                    if not a.dry_run and arm == HYB:
                        tmp = p + ".tmp"; open(tmp, "w").write(hyb_script(name, d, tier)); os.chmod(tmp, 0o755); os.replace(tmp, p)
                    elif not a.dry_run:
                        tmp = p + ".tmp"; json.dump(d, open(tmp, "w"), indent=1); os.replace(tmp, p)   # atomic: the worker never sees a half-written chunk
                    made.append(p)
    print("[queue] %s: %d tasks in the frozen matrix, %d frozen already, %d queued before, %d new tasks -> %d new chunks%s" % (q, tot["tasks"], tot["done"], tot["queued"], tot["new"], tot["chunks"], " (DRY RUN, nothing written)" if a.dry_run else ""))
    for p in made: print("  ", os.path.relpath(p, q))
    return 0
if __name__ == "__main__": sys.exit(main())
