#!/usr/bin/env python3
"""[10/19 audit] Validation of the spec_early fps correction (ours_to_timing.spec_early_dit_fields): for every clip under runs/ours_sf24_<set>_dit/
(re-run with STREAM_DECODE=0 = no side-stream decode at all in the chunk timer, gen/queue/8/302) compare its measured steady_fps_gen / gen_s with the
FROZEN clip's corrected steady_fps_gen_dit / gen_s_dit (runs/ours_sf24_<set>/) and with the frozen uncorrected steady_fps_gen / gen_s.  Writes
runs/ours_sf24_dit_CHECK.json = {clips: {task: {set, frozen_fps, frozen_fps_dit, dit_run_fps, frozen_gen_s, frozen_gen_s_dit, dit_run_gen_s,
ratio_dit_run_over_corrected, ratio_dit_run_over_frozen, mean_steps_frozen, mean_steps_dit_run}}, summary per set, verdict}.
Plain python3 (json), no GPU.  Idempotent; missing _dit clips are listed, not an error.
usage: python3 -I /home/hk28965/specwm/baselines/gen/dit_check.py [--runs DIR]"""
import os, sys, json, glob, argparse
B = "/home/hk28965/specwm/baselines"
ap = argparse.ArgumentParser(); ap.add_argument("--runs", default=B + "/runs"); a = ap.parse_args()
def num(x):
    try: return float(x)
    except Exception: return None
def r(x, y): return round(x / y, 3) if (x is not None and y) else None
clips, missing = {}, []
for d in sorted(glob.glob(a.runs + "/ours_sf24_*_dit")):
    set_ = os.path.basename(d)[len("ours_sf24_"):-len("_dit")]
    for tj in sorted(glob.glob(d + "/*/timing.json")):
        task = os.path.basename(os.path.dirname(tj)); ft = "%s/ours_sf24_%s/%s/timing.json" % (a.runs, set_, task)
        if not os.path.isfile(ft): missing.append(task + " (no frozen clip)"); continue
        D, F = json.load(open(tj)), json.load(open(ft))
        clips[task] = dict(set=set_, frozen_fps=num(F.get("steady_fps_gen")), frozen_fps_dit=num(F.get("steady_fps_gen_dit")), dit_run_fps=num(D.get("steady_fps_gen")),
                           frozen_gen_s=num(F.get("gen_s")), frozen_gen_s_dit=num(F.get("gen_s_dit")), dit_run_gen_s=num(D.get("gen_s")), dit_run_gen_s_dit=num(D.get("gen_s_dit")),
                           mean_steps_frozen=num(F.get("mean_steps")), mean_steps_dit_run=num(D.get("mean_steps")), dit_run_stream_decode=D.get("vae"),
                           frozen_chunk_s=F.get("chunk_s"), dit_run_chunk_s=D.get("chunk_s"))
        c = clips[task]; c["ratio_dit_run_over_corrected"] = r(c["dit_run_fps"], c["frozen_fps_dit"]); c["ratio_dit_run_over_frozen"] = r(c["dit_run_fps"], c["frozen_fps"])
        c["residual_gap_s_per_chunk"] = round((c["frozen_gen_s_dit"] - c["dit_run_gen_s"]) / max(1, len(F.get("chunk_s") or [1])), 4) if (c["frozen_gen_s_dit"] is not None and c["dit_run_gen_s"] is not None) else None
summ = {}
for s in sorted(set(c["set"] for c in clips.values())):
    cs = [c for c in clips.values() if c["set"] == s]; rc = [c["ratio_dit_run_over_corrected"] for c in cs if c["ratio_dit_run_over_corrected"]]; rf = [c["ratio_dit_run_over_frozen"] for c in cs if c["ratio_dit_run_over_frozen"]]
    summ[s] = dict(n=len(cs), mean_ratio_dit_run_over_corrected=round(sum(rc) / len(rc), 3) if rc else None, mean_ratio_dit_run_over_frozen=round(sum(rf) / len(rf), 3) if rf else None,
                   mean_steps_delta=round(sum((c["mean_steps_dit_run"] or 0) - (c["mean_steps_frozen"] or 0) for c in cs) / len(cs), 4) if cs else None)
allr = [c["ratio_dit_run_over_corrected"] for c in clips.values() if c["ratio_dit_run_over_corrected"]]
verdict = ("no _dit clips yet" if not allr else
           "ratio dit_run / corrected mean %.3f over %d clips: %s" % (sum(allr) / len(allr), len(allr),
           "the correction reproduces the decode-free fps within 5 %" if abs(sum(allr) / len(allr) - 1) <= 0.05 else
           ("the corrected fps is still CONSERVATIVE (decode-free run faster): the overlapped decode's GPU time inside 'denoise' is the remainder" if sum(allr) / len(allr) > 1.05 else
            "the corrected fps OVERSHOOTS the decode-free run: check mean_steps / GPU contention of the _dit run before using steady_fps_gen_dit")))
out = dict(clips=clips, summary=summ, missing=missing, verdict=verdict, note="[10/19 audit] dit_run = runs/ours_sf24_<set>_dit (STREAM_DECODE=0, RUNS_SUFFIX=_dit, exclusively held GPU); frozen = runs/ours_sf24_<set> (--stream-decode 1, shared GPUs at the time); frozen_fps_dit = steady_fps_gen x gen_s / (times.denoise + commit + verify). mean_steps may differ slightly between runs (early-exit head on a fresh run; same seed) - compare steps too.")
json.dump(out, open(a.runs + "/ours_sf24_dit_CHECK.json", "w"), indent=1)
print("[dit_check] %d clips, %d missing -> %s/ours_sf24_dit_CHECK.json" % (len(clips), len(missing), a.runs)); print("[dit_check]", verdict)
for t, c in sorted(clips.items()): print("  %-45s frozen %5s  corrected %5s  dit_run %5s  ratio %s  steps %s -> %s" % (t, c["frozen_fps"], c["frozen_fps_dit"], c["dit_run_fps"], c["ratio_dit_run_over_corrected"], c["mean_steps_frozen"], c["mean_steps_dit_run"]))
