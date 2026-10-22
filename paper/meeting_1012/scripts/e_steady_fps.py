"""Steady output fps for table (e) of notes/s2_item1_latency.md (Chenfeng-method streaming runs, jar + sandcastle x seeds 1234/4321/9999).
Per clip: ready time of the last frame of each chunk (loop.json frame_ready = [frame, chunk, t]); period = np.diff over chunks;
steady period = median of the periods of chunks >= 6 (np.diff(...)[5:], same chunk range as scripts/bench43_report.py for ours);
row value = median over the clips; steady fps = 12 / that. Decode is streamed on the same GPU, so it is inside these periods.
Also lists summary.json fps_mean (whole clip, (frames - 1) / (last - first ready)) and the ratio. CPU only.
Rows: profile (sdvg_profile/fast0 = original stack, fast1 = optimized stack; 6 clips, warm-up seed 1 skipped) and, added 10/12,
the quiet timing pass (timing_pass/latency/<arm>/summary.json per_clip dirs = the clips that summary used: 5 for ours704 / ours832 /
c704_sep, castle s1234 excluded for its cold compile; 6 for the others), keys tp_<arm>."""
import json, glob, os, numpy as np
ROOT = "/home/hk28965/specwm"
S = ROOT + "/lingbot-world-v2/specwm_out/sdvg_profile"
TP = ROOT + "/lingbot-world-v2/specwm_out/timing_pass"
OUT = ROOT + "/paper/meeting_1012/data/e_steady_fps.json"


def steady(d):
    last = {}
    for f, c, t in json.load(open(d + "/loop.json"))["frame_ready"]: last[c] = max(last.get(c, -1e18), t)
    p = np.diff([last[c] for c in sorted(last)])
    return float(np.median(p[5:]))


def row(dirs, sm):
    per = [steady(d) for d in dirs]
    m = float(np.median(per))
    return dict(clips=len(per), steady_period_s=round(m, 4), steady_fps=round(12 / m, 3), per_clip_period_s=[round(x, 4) for x in per],
                fps_mean_whole_clip=round(sm["fps_mean"], 3), whole_over_steady=round(sm["fps_mean"] / (12 / m), 3))


R = {}
for stack, fd in (("orig", "fast0"), ("opt", "fast1")):
    for arm, pre in (("draft", "none"), ("target", "target"), ("sdvg07", "ir_-0.7"), ("sdvg22", "ir_-2.2")):
        dirs = [d for d in sorted(glob.glob("%s/%s/*/%s_s*" % (S, fd, pre))) if not d.endswith("_s1")]  # _s1 = warm-up
        lab = "%s_%s" % (stack, arm)
        R[lab] = row(dirs, json.load(open("%s/latency/%s/summary.json" % (S, lab))))
for arm in ("ours704", "ours832", "c704_fused", "c704_sep", "u2_832_sep", "c576_pc1"):
    sm = json.load(open("%s/latency/%s/summary.json" % (TP, arm)))
    R["tp_" + arm] = row([os.path.join(ROOT, c["dir"]) for c in sm["per_clip"]], sm)
for lab, r in R.items():
    print(lab, r["clips"], r["steady_period_s"], r["steady_fps"], r["fps_mean_whole_clip"], r["whole_over_steady"])
json.dump(R, open(OUT, "w"), indent=1)
