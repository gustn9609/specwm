"""10/5 fixed-direction latent drift critic vs the other SDVG arms (loop1005, same tasks/seeds/GPU type).
Per clip: gen time, accepted chunks, plan, and frame distance to the same-seed target-only video (mean |pixel diff| over frames, 0-255; the
rollout is chaotic so this is a coarse 'stays near the 14B trajectory' signal, not per-seed fidelity). usage: python scripts/critfix_compare.py"""
import json, os, glob, numpy as np, imageio.v3 as iio
O = "/home/hk28965/specwm/lingbot-world-v2/specwm_out"
ARMS = [("draft only", "loop1005", "none"), ("SDVG ImageReward τ −0.7", "loop1005", "ir_-0.7"),
        ("latent critic τ 0.05 (bug: inverted)", "loop1005", "critic_noprev_noplanrc_0.05"), ("latent critic τ −3.87 (bug: inverted)", "loop1005", "critic_noprev_noplanrc_-3.87"),
        ("latent critic τ 0.05 (fixed)", "loop1005fix", "critic_noprev_noplanrc_0.05"), ("latent critic τ −3.87 (fixed)", "loop1005fix", "critic_noprev_noplanrc_-3.87"),
        ("target only", "loop1005", "target")]
TASKS = [("jar", "sand_filling_up_jar"), ("castle", "sandcastle_wave_impact")]; SEEDS = [1234, 4321, 9999]
def vid(p): return iio.imread(p, plugin="pyav").astype(np.float32)
rows = []
for short, task in TASKS:
    for s in SEEDS:
        ref = vid("%s/loop1005/%s/target_s%d/video.mp4" % (O, task, s))
        for name, root, arm in ARMS:
            d = "%s/%s/%s/%s_s%d" % (O, root, task, arm, s)
            if not os.path.exists(d + "/loop.json"): continue
            L = json.load(open(d + "/loop.json")); v = vid(d + "/video.mp4"); n = min(len(v), len(ref))
            dist = float(np.abs(v[:n] - ref[:n]).mean())
            rows.append(dict(scene=short, seed=s, arm=name, gen=L["gen_s"], acc=L["accepted"], plan=L["plan"], dist=dist, times=L["times"],
                             scores=[None if x is None else round(x, 2) for x in L["scores"]]))
json.dump(rows, open(O + "/loop1005fix/compare.json", "w"), indent=1)
print("| arm | latent gen ↓ (s) | accepted ↑ (of 12) | frame dist to target-only ↓ |"); print("|---|---|---|---|")
for name, _, _ in ARMS:
    R = [r for r in rows if r["arm"] == name]
    if R: print("| %s | %.1f | %.1f | %.1f |" % (name, np.mean([r["gen"] for r in R]), np.mean([r["acc"] for r in R]), np.mean([r["dist"] for r in R])))
print()
for r in rows:
    if "fixed" in r["arm"]: print(r["scene"], r["seed"], r["arm"], r["plan"], r["gen"], r["scores"])
