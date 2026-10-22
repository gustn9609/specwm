"""Majority-vote blind ratings (3 raters) per clip and summarise per (task, arm): clean count, unanimous count, top drift categories.
usage: python blind_majority.py DIR   (reads DIR/mapping.json and DIR/r*_p*.json)"""
import sys, json, glob, collections
D = sys.argv[1]; mp = json.load(open(D + "/mapping.json")); V = collections.defaultdict(list)
for f in sorted(glob.glob(D + "/r*_p*.json")):
    for r in json.load(open(f)): V[r["id"]].append(r)
G = collections.defaultdict(lambda: dict(n=0, clean=0, unan_clean=0, cats=collections.Counter(), seeds_clean=[]))
for hid, rs in V.items():
    m = mp[hid]; g = G[(m["task"], m["arm"])]; dv = sum(r["drift"] for r in rs); clean = dv * 2 < len(rs)
    g["n"] += 1; g["clean"] += clean; g["unan_clean"] += dv == 0
    if clean: g["seeds_clean"].append(m["seed"])
    for r in rs: g["cats"].update(r.get("categories", []))
for (t, a), g in sorted(G.items()):
    print("%-24s %-11s clean %d/%d (unanimous %d)  seeds %s  cats %s" % (t, a, g["clean"], g["n"], g["unan_clean"], sorted(g["seeds_clean"]), dict(g["cats"].most_common(4))))
