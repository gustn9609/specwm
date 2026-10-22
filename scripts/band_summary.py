"""Summarise --band-probe runs: per node, median over chunks of (a) share of the gap energy in each low-frequency subspace, (b) log10 of the exact
acceptance of a whole latent frame / whole chunk when the test is restricted to that subspace. usage: band_summary.py [RUN_DIR ...]"""
import sys, json, glob, numpy as np
runs = sys.argv[1:] or sorted(glob.glob("/home/hk28965/specwm/lingbot-world-v2/specwm_out/spec_v0/*/bandprobe_*_s*"))
K = ["full", "k2", "k4", "k8", "k16", "k0"]; NAME = {"full": "full res (58x106)", "k2": "2x2 blocks (29x53)", "k4": "4x4 (15x27)", "k8": "8x8 (8x14)", "k16": "16x16 (4x7)", "k0": "1 value per channel"}
out = {}
for r in runs:
    d = json.load(open(r + "/spec_v0.json")); tag = r.split("/")[-2][:6] + " " + r.split("/")[-1]
    for node in ("0", "1", "2"):
        L = [l["band"][node] for l in d["log"] if "band" in l and node in l["band"]]
        if not L: continue
        row = {}
        for k in K:
            share = np.median([(x[k]["norm"][-1] / x["full"]["norm"][-1]) ** 2 for x in L]); fr = np.median([np.median(x[k]["log10acc_frame"]) for x in L]); ch = np.median([x[k]["log10acc_chunk"] for x in L])
            row[k] = (share, fr, ch)
        out[(tag, node)] = row
        print("\n%s  node %s  (%d chunks; gap norm per chunk median %.1f)" % (tag, node, len(L), np.median([x["full"]["norm"][-1] for x in L])))
        print("  %-22s %-14s %-24s %s" % ("subspace", "gap energy", "log10 P(accept frame)", "log10 P(accept chunk)"))
        for k in K: s, f, c = row[k]; print("  %-22s %-14s %-24s %s" % (NAME[k], "%.1f%%" % (100 * s), "%.1f" % f, "%.1f" % c))
json.dump({"%s|%s" % k: v for k, v in out.items()}, open("/home/hk28965/specwm/exp/kveval/band_summary.json", "w"), indent=1)
