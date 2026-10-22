#!/usr/bin/env python3
"""[10/17 frozen baselines] baselines/TABLE.md from runs/<model_arm>/<set>__<scene>__<P>/timing.json + metrics/{vlm,vbench6,rpe,reaction}.json (PROTOCOL
section 4). Table 1, per model_arm x scene set: n clips, output resolution, steady fps (DiT only / end-to-end incl. decode), TTFF, peak VRAM allocated,
judge quality (1-10) / drift (0-100), VBench quality score / imaging (MUSIQ, 0-100) / MUSIQ drift, RPE Rot vs the commanded program, P5 reaction (s and
frames; mean over the clip's valid switches). Cells are means over the arm's clips of that set; '(k)' = the metric exists on only k of the n clips.
Per set and column: **bold** = best, <u>underline</u> = second (direction in the header). Table 2: paired delta vs the reference arm (--ref, default
ours_teacher: the ref arm of a clip is runs/<ref>_<set><suffix>, e.g. ours_teacher_wp for waypoint/wp__*, ours_teacher_neutral_832_nocap for
astra_nocap/neutral_832__*_nocap) on the same (set, scene, program, suffix): mean(arm - ref), the better / equal / worse clip counts (direction-aware,
'equal' within the tolerance in the header) and the e2e fps ratio arm / ref. [10/17] The 'set' column of every table is <set><suffix>: the empty-prompt
arms (suffix _nocap) form their own group, so they are ranked and paired only against arms of the same suffix. Per-clip rows go to --csv (default
<first metrics dir>/clips.csv).
[10/19 audit] additions (PROTOCOL section 8): (F1) --metrics takes a comma list of dirs, merged (later dir wins on a conflict, counts printed);
(F2) abot / neutral_abot set means use programs P1-P5 only for EVERY arm (ABot's P2p/P4p are byte-identical to P2/P4); ours' P2p/P4p go to separate
'(P2p+P4p only)' rows; (F3) abot set gets '_1st scenes only' / '_3rd scenes' split rows per arm; (F4) metrics/luminance.json -> column 'black' (count of
clips that go black) + 'excl. black' rows; (F5) astra_nocap labelled 'Astra on neutral scenes, no caption'; (F6) n expected from tasks/<set>.json,
'(partial)' when short; (F7) 'GPUs' column (timing.json gpus, default 1), VRAM of 2-GPU hybrid arms = device GPU peak '(dev)'; (F8) gen/idle_timing_map.json
re-measures override the listed timing fields, steady_fps_gen_dit / gen_s_dit override fps gen / gen s; (F9) Waypoint speed columns footnoted as a
shared-GPU lower bound until the _idle re-measure exists; (F10) <set>_long (P6) clips stay in their own <set>_long group. Table 1 is printed per set
with its footnotes; extra rows (split / P2p+P4p / excl. black) are never ranked.
usage: python3 /home/hk28965/specwm/baselines/report.py [--runs DIR] [--metrics DIR[,DIR2,...]] [--out FILE] [--csv FILE] [--ref ours_teacher] [--idle-map FILE]   (plain python3, no numpy)"""
import os, re, sys, json, glob, argparse, time, math
B = "/home/hk28965/specwm/baselines"
ap = argparse.ArgumentParser(); ap.add_argument("--runs", default=B + "/runs"); ap.add_argument("--metrics", default=B + "/metrics", help="[10/19 audit] comma list of metrics dirs, merged; later dir wins on conflict"); ap.add_argument("--out", default=B + "/TABLE.md")
ap.add_argument("--csv", default="", help="per-clip csv (default <first metrics dir>/clips.csv)"); ap.add_argument("--ref", default="ours_teacher", help="reference arm (exact name or <ref>_<set>)")
ap.add_argument("--idle-map", default=B + "/gen/idle_timing_map.json", help="[10/19 audit] {arm: {runs_dir, fields}} exclusive-GPU timing re-measures")
a = ap.parse_args()
a.mdirs = [(d if os.path.isabs(d) else (os.path.abspath(d) if os.path.isdir(d) else os.path.join(B, d))) for d in a.metrics.split(",") if d]   # [10/19 audit] F1
a.csv = a.csv or os.path.join(a.mdirs[0], "clips.csv")
# key, header, direction (+1 higher better, -1 lower better, 0 none), format, tolerance for 'equal' in the paired table (None = not paired)
COLS = [("n", "n", 0, "%d", None), ("res", "res", 0, "%s", None), ("gpus", "GPUs", 0, "%d", None), ("fps_gen", "fps gen ↑", 1, "%.1f", None), ("fps_e2e", "fps e2e ↑", 1, "%.1f", None),
        ("gen_s", "gen s ↓", -1, "%.1f", None), ("ttff_s", "TTFF s ↓", -1, "%.2f", None), ("vram", "VRAM GiB ↓", -1, "%.1f", None), ("judge_q", "judge Q ↑", 1, "%.2f", 0.0), ("judge_drift", "judge drift ↓", -1, "%.1f", 0.0),
        ("vb_q", "VBench Q ↑", 1, "%.2f", 0.5), ("vb_img", "imaging ↑", 1, "%.1f", 0.5), ("musiq_drift", "MUSIQ drift ↓", -1, "%.2f", 0.1), ("rpe_rot", "RPE Rot ° ↓", -1, "%.3f", 0.1),
        ("react_s", "react s ↓", -1, "%.2f", 0.05), ("react_fr", "react fr ↓", -1, "%.1f", None), ("black", "black", 0, "%d", None)]   # [10/19 audit] + gpus, gen s, black
PAIRED = [c for c in COLS if c[4] is not None]
NOMEAN = ("n", "res", "gpus", "black")   # [10/19 audit] columns that are not clip means
ABOT_SETS = {"abot", "neutral_abot"}; FIVE = ("P1", "P2", "P3", "P4", "P5"); DUP = ("P2p", "P4p")   # [10/19 audit] F2
def is_abot_arm(arm): return arm.split("_")[0] == "abot"   # abot, abot_t5offload: no key magnitude -> P2p = P2, P4p = P4
def num(x):
    try: v = float(x); return None if math.isnan(v) else v
    except Exception: return None
def mean(xs): xs = [x for x in xs if x is not None]; return (sum(xs) / len(xs), len(xs)) if xs else (None, 0)
def load_metrics(path):
    d = json.load(open(path)) if os.path.exists(path) else {}; out = {}
    for k, v in d.items():
        if isinstance(v, dict) and k.startswith("/"): out[k] = v; out[os.path.realpath(k)] = v
    return out
def same(x, y, tol=1e-4):   # [10/19 audit] F1: numeric leaves equal within tol (relative), everything else exact
    if isinstance(x, dict) and isinstance(y, dict): return x.keys() == y.keys() and all(same(x[k], y[k], tol) for k in x)
    if isinstance(x, list) and isinstance(y, list): return len(x) == len(y) and all(same(p, q, tol) for p, q in zip(x, y))
    if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool) and not isinstance(y, bool): return abs(x - y) <= tol * max(1.0, abs(x), abs(y))
    return x == y
def load_merged(name):   # [10/19 audit] F1: merge <dir>/<name>.json over a.mdirs; later dir wins; count overlaps / value conflicts
    out, seen, overlap, conflict = {}, {}, 0, 0
    for d in a.mdirs:
        p = os.path.join(d, name + ".json"); m = json.load(open(p)) if os.path.exists(p) else {}
        for k, v in m.items():
            if not (isinstance(v, dict) and k.startswith("/")): continue
            if k in seen and seen[k] != d:
                overlap += 1
                if not same(out[k], v): conflict += 1
            out[k] = v; out[os.path.realpath(k)] = v; seen[k] = d
    if len(a.mdirs) > 1: print("[report] %s: %d keys from %d dirs, %d overlapping, %d conflicting beyond 1e-4 (later dir wins)" % (name, len(seen), len(a.mdirs), overlap, conflict))
    return out
def lookup(m, vid): return m.get(vid) or m.get(os.path.abspath(vid)) or m.get(os.path.realpath(vid))
def parse_clip(name):   # <set>__<scene>__P<id>[p][suffix] -> (set, scene, P, suffix); [10/17 verifier fix] P2p / P4p are programs, not a suffix 'p'; [10/19] P6_long -> P6 + suffix _long (F10)
    parts = name.split("__"); m = re.match(r"(P\d+p?)(.*)$", parts[-1]) if len(parts) >= 2 else None
    if not m: return None
    return (parts[0] if len(parts) >= 3 else "?"), "__".join(parts[1:-1]) or parts[0], m[1], m[2]
VLM, VB, RPE, REA, LUM = (load_merged(f) for f in ("vlm", "vbench6", "rpe", "reaction", "luminance"))
def load_idle_map(path):   # [10/19 audit] F8: {arm: {runs_dir, fields}} or the gen/idle_timing_map.json schema {"remeasures": [{frozen_dir, prefer_dir, fields, ...}]} (arm = basename of frozen_dir; prefer_dir null = validation only, skipped)
    if not os.path.exists(path): return {}
    try: m = json.load(open(path))
    except Exception as e: print("[report] bad idle map", path, e, file=sys.stderr); return {}
    out = {}
    for arm, v in m.items():
        if isinstance(v, dict) and "runs_dir" in v: out[arm] = dict(runs_dir=v["runs_dir"] if os.path.isabs(v["runs_dir"]) else os.path.join(B, v["runs_dir"]), fields=list(v.get("fields", [])))
    for e in (m.get("remeasures") or []):
        if not (isinstance(e, dict) and e.get("prefer_dir") and e.get("frozen_dir")): continue
        pd = e["prefer_dir"] if os.path.isabs(e["prefer_dir"]) else os.path.join(B, e["prefer_dir"])
        out[os.path.basename(e["frozen_dir"].rstrip("/"))] = dict(runs_dir=pd, fields=list(e.get("fields", [])))
    return out
IDLE = load_idle_map(a.idle_map)
if IDLE: print("[report] idle timing map: %s -> %s" % (a.idle_map, ", ".join("%s <- %s%s" % (k, os.path.relpath(v["runs_dir"], B), "" if os.path.isdir(v["runs_dir"]) else " (absent yet)") for k, v in sorted(IDLE.items()))))
rows = []
for arm_dir in sorted(glob.glob(os.path.join(a.runs, "*"))):
    if not os.path.isdir(arm_dir): continue
    arm = os.path.basename(arm_dir)
    for cd in sorted(glob.glob(os.path.join(arm_dir, "*"))):
        vid, tj = os.path.join(cd, "video.mp4"), os.path.join(cd, "timing.json"); p = parse_clip(os.path.basename(cd))
        if not (os.path.isfile(vid) and os.path.isfile(tj) and p): continue
        try: T = json.load(open(tj))
        except Exception as e: print("[report] bad timing.json", tj, e, file=sys.stderr); continue
        idle = ""   # [10/19 audit] F8: exclusive-GPU re-measure overrides the listed fields when <runs_dir>/<clip>/timing.json exists
        if arm in IDLE and isinstance(IDLE[arm], dict):
            tji = os.path.join(IDLE[arm].get("runs_dir", ""), os.path.basename(cd), "timing.json")
            if os.path.isfile(tji):
                try:
                    Ti = json.load(open(tji))
                    for f in IDLE[arm].get("fields", []):
                        if f in Ti: T[f] = Ti[f]
                    idle = IDLE[arm].get("runs_dir", "")
                except Exception as e: print("[report] bad idle timing.json", tji, e, file=sys.stderr)
        res = T.get("resolution_out"); res = "x".join(str(x) for x in res) if isinstance(res, (list, tuple)) else str(res or "")
        dit = "steady_fps_gen_dit" in T or "gen_s_dit" in T   # [10/19 audit] F8: spec_early arms, DiT-only numbers corrected for the side-stream decode overlap
        shared = "shared" in str(T.get("timing_caveat", "")).lower() or "shared gpu" in str(T.get("notes", "")).lower()   # [10/19 audit] F9
        r = dict(arm=arm, set=p[0], scene=p[1], P=p[2], suffix=p[3], setx=p[0] + p[3], clip=os.path.basename(cd), video=vid, res=res,
                 fps_gen=num(T.get("steady_fps_gen_dit", T.get("steady_fps_gen"))), fps_e2e=num(T.get("steady_fps_e2e")), gen_s=num(T.get("gen_s_dit", T.get("gen_s"))),
                 ttff_s=num(T.get("ttff_s")), vram=num(T.get("peak_alloc_GiB", T.get("peak_GiB"))), frames=T.get("frames"), fps_native=T.get("fps_native"), model=T.get("model", ""),
                 gpus=int(num(T.get("gpus")) or 1), idle=idle, dit=dit, shared=shared)
        v = lookup(VLM, vid) or {}; r["judge_q"] = num(v.get("quality")); r["judge_drift"] = num(v.get("drift"))
        v = lookup(VB, vid) or {}; r["vb_q"] = num(v.get("quality_score")); r["vb_img"] = None if num(v.get("imaging_quality")) is None else 100 * num(v.get("imaging_quality")); r["musiq_drift"] = num(v.get("musiq_drift"))
        v = lookup(RPE, vid) or {}; r["rpe_rot"] = num(v.get("rot")); r["rpe_trans"] = num(v.get("trans"))
        v = lookup(REA, vid) or {}; r["react_s"] = num(v.get("mean_react_s")); r["react_fr"] = num(v.get("mean_react_frames"))
        v = lookup(LUM, vid) or {}; r["black"] = (None if "black" not in v else bool(v["black"])); r["lum_first"] = num(v.get("first_s")); r["lum_last"] = num(v.get("last_s"))   # [10/19 audit] F4
        rows.append(r)
if not rows: print("[report] no clips under", a.runs); sys.exit(1)
# ---------------------------------------------------------------- table 1: per arm x set (set = <set><suffix>: *_nocap / *_long clips are their own group)
def task_ids(setx):   # [10/19 audit] F6: the frozen task list of a group -> [(scene, P)]
    p = os.path.join(B, "tasks", setx + ".json")
    if not os.path.isfile(p): return None
    try: ids = json.load(open(p))
    except Exception: return None
    out = []
    for t in ids:
        q = parse_clip(t if isinstance(t, str) else str(t.get("id", "")))
        if q: out.append((q[1], q[2]))
    return out
def expected_n(setx, progs=None, scene_ok=None):
    ids = task_ids(setx)
    if ids is None: return None
    return sum(1 for s, p in ids if (progs is None or p in progs) and (scene_ok is None or scene_ok(s)))
def aggregate(rs):
    g = dict(n=len(rs), res="/".join(sorted({r["res"] for r in rs if r["res"]})) or "-", gpus=max(r["gpus"] for r in rs), black=sum(1 for r in rs if r["black"]), black_k=sum(1 for r in rs if r["black"] is not None))
    for k, _, d, _, _ in COLS:
        if k in NOMEAN: continue
        g[k], g[k + "_k"] = mean([r[k] for r in rs])
    return g
groups = {}
for r in rows: groups.setdefault((r["arm"], r["setx"]), []).append(r)
def set_rows(rs, setx):   # [10/19 audit] F2: the set-mean clips of a group
    return [r for r in rs if r["P"] in FIVE] if setx in ABOT_SETS else rs
agg = {}
for (arm, st), rs in groups.items():
    rs2 = set_rows(rs, st)
    if rs2: agg[(arm, st)] = aggregate(rs2)
def fmt(k, f, g, mark="", exp=None):
    v = g.get(k)
    if v is None: return "-"
    s = f % v
    if k == "n" and exp is not None and v != exp: s += " (partial)" if v < exp else " (extra)"   # [10/19 audit] F6
    if k == "black":   # luminance known on only k clips ('-' when on none)
        if not g.get("black_k"): return "-"
        if g["black_k"] < g["n"]: s += " (%d)" % g["black_k"]
    if k == "vram" and g.get("gpus", 1) > 1: s += " (dev)"   # [10/19 audit] F7: hybrid arms, device GPU peak only
    if k not in NOMEAN and g.get(k + "_k", g["n"]) < g["n"]: s += " (%d)" % g[k + "_k"]
    return "**%s**" % s if mark == "best" else ("<u>%s</u>" % s if mark == "second" else s)
sets = sorted({st for _, st in agg}); marks = {}
for st in sets:
    arms = [arm for (arm, s2) in agg if s2 == st]
    for k, _, d, _, _ in COLS:
        if d == 0: continue
        vals = sorted([(agg[(arm, st)][k], arm) for arm in arms if agg[(arm, st)].get(k) is not None], key=lambda x: -x[0] * d)
        if len(vals) < 2: continue
        f = dict((kk, ff) for kk, _, _, ff, _ in COLS)[k]; levels = []   # ties at the printed precision share a rank
        for v, arm in vals:
            if not levels or f % v != levels[-1][0]: levels.append((f % v, []))
            levels[-1][1].append(arm)
        for arm in levels[0][1]: marks[(arm, st, k)] = "best"
        if len(levels) > 1:
            for arm in levels[1][1]: marks[(arm, st, k)] = "second"
def label(arm): return "astra_nocap (Astra on neutral scenes, no caption)" if arm == "astra_nocap" else arm   # [10/19 audit] F5
L = ["# Frozen baseline table (generated %s by baselines/report.py)" % time.strftime("%Y-%m-%d %H:%M %Z"), "",
     "runs: `%s` (%d arms, %d clips); metrics: `%s` (judge %d, VBench %d, RPE %d, reaction %d, luminance %d clips with values)." % (a.runs, len({r["arm"] for r in rows}), len(rows), ", ".join(a.mdirs),
     sum(r["judge_q"] is not None for r in rows), sum(r["vb_q"] is not None for r in rows), sum(r["rpe_rot"] is not None for r in rows), sum(r["react_s"] is not None for r in rows), sum(r["black"] is not None for r in rows)), "",
     "Columns: GPUs = GPUs the arm runs on (timing.json `gpus`, default 1); fps gen = steady DiT-only fps, fps e2e = steady fps incl. decode, gen s = DiT-only generation time of the 10-s clip, TTFF = input to first decoded frame (s), "
     "VRAM = peak allocated GiB (timing.json, PROTOCOL section 4; '(dev)' = device GPU only for 2-GPU hybrid arms); "
     "judge Q (1-10) / drift (0-100) = Qwen3-VL clip judge on 8 evenly spaced frames; VBench Q = 6-dim quality score (0-100, clips resampled to 16 fps), imaging = MUSIQ (0-100), "
     "MUSIQ drift = std of the 2 s segment means; RPE Rot = mean relative rotation error (deg per 1/8 s step) of DA3-recovered poses vs the commanded program (Sim(3)-aligned per fifth); "
     "react = P5 reaction time from the model's effective input switch until the optical-flow sign follows (s / frames at native fps, mean over valid switches); "
     "black = number of the row's clips that go black (last-second mean luminance of the 16x9 gray downscale < 10 or < 30 % of the first second; metrics/luminance.json). "
     "set = <scene set><suffix> of the clip names (suffix _nocap = empty-prompt arm, _long = 60-s P6 clips: each ranked and paired only within its own suffix). "
     "n = clips in the row; '(partial)' = fewer than the frozen task list tasks/<set>.json expects (generation or scoring still running). "
     "Arrows give the direction; per scene set, **bold** = best arm, <u>underline</u> = second among the main rows; '(k)' = metric present on k of n clips. Means over clips. "
     "Extra rows (split by scene type, P2p+P4p only, excl. black) are shown in italics and are never ranked.", "",
     "## 1. Per model arm x scene set (one table per set, footnotes below each)", ""]
HDR = ["| model_arm | set | " + " | ".join(h for _, h, _, _, _ in COLS) + " |", "|---|---|" + "|".join("---:" for _ in COLS) + "|"]
def row_line(name, st, g, arm=None, exp=None, italic=False):
    cells = " | ".join(fmt(k, f, g, marks.get((arm, st, k), "") if arm else "", exp if k == "n" else None) for k, _, _, f, _ in COLS)
    return "| %s | %s | %s |" % (("*%s*" % name) if italic else name, st, cells)
for st in sets:
    progs = FIVE if st in ABOT_SETS else None; exp = expected_n(st, progs); notes = []; cat = {}   # cat[category] = [arm, ...] -> one footnote per category
    def note(c, arm): cat.setdefault(c, []).append(arm)
    L += ["### set `%s`" % st + ("" if exp is None else " (n expected %d per arm%s)" % (exp, " = 5 programs P1-P5" if st in ABOT_SETS else ("" if not st.endswith("_long") else " = 4 scenes x P6"))), ""] + HDR
    for arm in sorted(arm for (arm, s2) in agg if s2 == st):
        rs_all = groups[(arm, st)]; rs = set_rows(rs_all, st); g = agg[(arm, st)]
        L.append(row_line(label(arm), st, g, arm, exp))
        if st == "abot":   # [10/19 audit] F3: first- vs third-person scenes of the ABot set
            for suf, nm in (("_1st", "_1st scenes only"), ("_3rd", "_3rd scenes")):
                sub = [r for r in rs if r["scene"].endswith(suf)]
                if sub: L.append(row_line("%s (%s, n=%d)" % (label(arm), nm, len(sub)), st, aggregate(sub), None, expected_n(st, progs, lambda s, suf=suf: s.endswith(suf)), True))
        if st in ABOT_SETS and not is_abot_arm(arm):   # [10/19 audit] F2: ours' P2p/P4p listed apart, never in the set mean
            sub = [r for r in rs_all if r["P"] in DUP]
            if sub: L.append(row_line("%s (P2p+P4p only, n=%d)" % (label(arm), len(sub)), st, aggregate(sub), None, expected_n(st, DUP), True))
        if g["black"]:   # [10/19 audit] F4
            sub = [r for r in rs if not r["black"]]
            if sub: L.append(row_line("%s excl. black (n=%d)" % (label(arm), len(sub)), st, aggregate(sub), None, None, True))
        # footnotes of this arm in this set
        if st in ABOT_SETS and is_abot_arm(arm) and any(r["P"] in DUP for r in rs_all):
            note("P2p/P4p clips exist but are byte-identical to the P2/P4 clips (no key magnitude) and are not counted", "`%s` (%d clips)" % (arm, sum(1 for r in rs_all if r["P"] in DUP)))
        if g["gpus"] > 1: note("2 GPUs (device + cloud): VRAM = device GPU peak only (the cloud GPU's peak is timing.json `cloud_peak_alloc_GiB`); TTFF / fps e2e are not measured in the two-process mode", "`%s`" % arm)
        ki = [r["idle"] for r in rs if r["idle"]]
        if ki: note("timing columns (%s) from an exclusive-GPU re-measure (%s)" % (", ".join(IDLE.get(arm, {}).get("fields", [])), sorted(set(ki))[0]), "`%s` (%d of %d clips)" % (arm, len(ki), len(rs)))
        kd = sum(1 for r in rs if r["dit"])
        if kd: note("fps gen / gen s are DiT-only, corrected for the side-stream decode overlap (audit 10/19)", "`%s` (%d of %d clips)" % (arm, kd, len(rs)))
        if arm.startswith("waypoint") and len(ki) < len(rs): note("speed columns (fps gen / fps e2e / TTFF / gen s) are a lower bound, shared-GPU measurement (timing.json `timing_caveat`); the `_idle` exclusive-GPU re-measure is pending", "`%s` (%d of %d clips)" % (arm, len(rs) - len(ki), len(rs)))   # [10/19 audit] F9
        elif not arm.startswith("waypoint") and len(ki) < len(rs):
            ks = sum(1 for r in rs if r["shared"] and not r["idle"])
            if ks: note("timing.json flags a shared-GPU measurement (speed columns are a lower bound there)", "`%s` (%d of %d clips)" % (arm, ks, len(rs)))
        if g["black"]: note("clips go black (see the 'excl. black' rows)", "`%s` (%d of %d)" % (arm, g["black"], g["n"]))
    for c, arms_c in cat.items(): notes.append("%s: %s." % (", ".join(arms_c), c))
    if st in ABOT_SETS: notes.insert(0, "Set means use programs P1-P5 only for EVERY arm (ABot has no key magnitude, so its P2p = P2 and P4p = P4 videos are identical); ours' P2p/P4p clips are listed in separate '(P2p+P4p only)' rows and are not ranked.")
    if st == "abot": notes.insert(1, "5 of the 12 ABot scenes are third-person (`_3rd`: aurora_ice, case4_desert_caravan, desert_camel, desert_valley, ruins_explorer), where WASD moves the character and not the camera: RPE Rot / reaction on them is not a camera comparison; see the '_1st scenes only' (7 scenes) and '_3rd scenes' (5 scenes) split rows (not ranked).")
    if st == "astra": notes.insert(0, "Astra's own-scene set = 3 scenes x 7 programs = 21 clips per arm (n expected 21).")
    if st == "neutral_832_nocap" and ("astra_nocap", st) in agg: notes.insert(0, "`astra_nocap` = Astra run on the 12 neutral_832 scenes with an empty prompt (NOT the Astra scenes without captions); it belongs to this group only and is ranked here.")
    if any(agg[(arm, s2)]["n"] != exp for (arm, s2) in agg if s2 == st) and exp is not None: notes.append("'(partial)' rows: n below the %d clips of tasks/%s.json (generation or scoring still running)." % (exp, st))
    L += [""] + ["- " + x for x in notes] + [""]
# ---------------------------------------------------------------- table 2: paired vs reference
def is_ref(arm, st, suffix=""): return arm == a.ref or arm == "%s_%s%s" % (a.ref, st, suffix)   # [10/17] the ref arm of a clip = <ref>_<set><suffix> (ours_teacher_neutral_832_nocap for *_nocap clips); an arm named exactly <ref> is the ref for everything
def key(r): return (r["set"], r["scene"], r["P"], r["suffix"])
ref = {}
for r in rows:
    if is_ref(r["arm"], r["set"], r["suffix"]): ref.setdefault(key(r), r)
L += ["", "## 2. Paired delta vs `%s_<set><suffix>` on the same (set, scene, program, suffix)" % a.ref, "",
      "delta = mean(arm - ref) over paired clips; b/e/w = clips where the arm is better / equal / worse (direction-aware; equal within tolerance: " +
      ", ".join("%s %s" % (h.split(" ↑")[0].split(" ↓")[0], ("exact" if t == 0 else "±%g" % t)) for _, h, _, _, t in PAIRED) + "); fps e2e x = mean per-clip ratio arm / ref. "
      "[10/19 audit] abot / neutral_abot pairs use programs P1-P5 only (ABot's P2p/P4p duplicate P2/P4).", ""]
if not ref: L.append("_no reference clips (no arm named `%s` / `%s_<set><suffix>` in runs/)_" % (a.ref, a.ref))
else:
    L += ["| model_arm | set | paired n | fps e2e x | " + " | ".join("Δ " + h + " (b/e/w)" for _, h, _, _, _ in PAIRED) + " |", "|---|---|---:|---:|" + "|".join("---:" for _ in PAIRED) + "|"]
    for (arm, st), rs in sorted(groups.items()):
        if all(is_ref(arm, r["set"], r["suffix"]) for r in rs): continue
        rs = set_rows(rs, st)   # [10/19 audit] F2
        pairs = [(r, ref[key(r)]) for r in rs if key(r) in ref and not is_ref(arm, r["set"], r["suffix"])]
        if not pairs: continue
        ratio, _ = mean([x["fps_e2e"] / y["fps_e2e"] for x, y in pairs if x["fps_e2e"] and y["fps_e2e"]]); cells = []
        for k, _, d, f, tol in PAIRED:
            pp = [(x[k], y[k]) for x, y in pairs if x[k] is not None and y[k] is not None]
            if not pp: cells.append("-"); continue
            dl = [x - y for x, y in pp]; b = sum(1 for z in dl if z * d > tol); w = sum(1 for z in dl if -z * d > tol); e = len(dl) - b - w
            cells.append(("%+" + f[1:]) % (sum(dl) / len(dl)) + " (%d/%d/%d)" % (b, e, w) + ("" if len(pp) == len(pairs) else " n=%d" % len(pp)))
        L.append("| %s | %s | %d | %s | " % (label(arm), st, len(pairs), "-" if ratio is None else "%.2fx" % ratio) + " | ".join(cells) + " |")
# ---------------------------------------------------------------- per-program RPE / reaction (control-following detail)
L += ["", "## 3. Control following per program (RPE Rot ° ↓ mean over sets; P5 reaction s ↓)", "",
      "[10/19 audit] ABot's P2p / P4p columns repeat its P2 / P4 (identical videos). 10-s programs only (P6 = `_long` groups are excluded).", ""]
progs = sorted({r["P"] for r in rows if r["suffix"] != "_long"}, key=lambda p: (int(re.match(r"P(\d+)", p)[1]), p)); arms = sorted({r["arm"] for r in rows})   # P1 < P2 < P2p < P3 ...
L += ["| model_arm | " + " | ".join("RPE %s" % p for p in progs) + " | react P5 s | react P5 switches found/valid |", "|---|" + "|".join("---:" for _ in progs) + "|---:|---:|"]
for arm in arms:
    cells = []
    for p in progs:
        v, k = mean([r["rpe_rot"] for r in rows if r["arm"] == arm and r["P"] == p and r["suffix"] != "_long"]); cells.append("-" if v is None else "%.3f (n=%d)" % (v, k))
    v, k = mean([r["react_s"] for r in rows if r["arm"] == arm and r["P"] == "P5"]); fr = [lookup(REA, r["video"]) for r in rows if r["arm"] == arm and r["P"] == "P5"]; fr = [x for x in fr if x]
    if not any(cells[i] != "-" for i in range(len(cells))) and v is None: continue
    L.append("| %s | " % label(arm) + " | ".join(cells) + " | %s | %s |" % ("-" if v is None else "%.2f (n=%d)" % (v, k), "-" if not fr else "%d/%d" % (sum(x.get("n_found", 0) for x in fr), sum(x.get("n_valid", 0) for x in fr))))
open(a.out, "w").write("\n".join(L) + "\n")
with open(a.csv, "w") as f:
    keys = ["arm", "set", "scene", "P", "suffix", "res", "gpus", "frames", "fps_native", "fps_gen", "fps_e2e", "gen_s", "ttff_s", "vram", "judge_q", "judge_drift", "vb_q", "vb_img", "musiq_drift", "rpe_rot", "rpe_trans", "react_s", "react_fr",
            "black", "lum_first", "lum_last", "idle", "dit", "shared", "video"]   # [10/19 audit] + gpus, gen_s, black, lum_*, idle, dit, shared
    f.write(",".join(keys) + "\n")
    for r in rows: f.write(",".join("" if r.get(k) is None else (("%.4f" % r[k]) if isinstance(r[k], float) else str(int(r[k]) if isinstance(r[k], bool) else r[k])) for k in keys) + "\n")
print("[report] %s: %d arms, %d clips, %d sets -> %s (+ %s)" % (a.runs, len(arms), len(rows), len(sets), a.out, a.csv))
