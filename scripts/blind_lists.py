"""Split a blind-sheet folder into rater lists: DIR/r{1,2,3}_p{0..N-1}.txt with 'id scene' pairs (same shuffled order for all raters).
usage: blind_lists.py DIR NPARTS [SEED]"""
import sys, json, random
B, n = sys.argv[1], int(sys.argv[2]); seed = int(sys.argv[3]) if len(sys.argv) > 3 else 29
m = json.load(open(B + "/mapping.json")); ids = sorted(m); random.Random(seed).shuffle(ids)
sc = lambda i: "jar" if "jar" in m[i]["task"] else "castle"
parts = [ids[i::n] for i in range(n)]
for r in (1, 2, 3):
    for p, part in enumerate(parts): open("%s/r%d_p%d.txt" % (B, r, p), "w").write(", ".join("%s %s" % (i, sc(i)) for i in part) + "\n")
print(len(ids), [len(p) for p in parts])
