"""[10/20] Re-derive chunk_s / draft_s / gen_s for the hybrid DEVICE streams converted after the 10/19 node-label edit of ours_to_timing.py (that edit
had turned `chunk_s=tc, draft_s=..., gen_s=...` into a comment, so runs/ours_<arm>_dev_<set>/<task>/timing.json lost those three fields; steady_fps_gen
was unaffected). Source of truth = the raw device timing.json named in each file's "source" (kept after --purge): chunk_s = per-chunk total_s,
draft_s = per-chunk draft_s, gen_s = sum(chunk_s). Idempotent; prints a per-dir count. usage: python3 -I gen/fix_dev_gen_s_1020.py [--dry]"""
import json, glob, os, sys, collections
dry = "--dry" in sys.argv; n = collections.Counter(); miss = []
for p in sorted(glob.glob("/home/hk28965/specwm/baselines/runs/ours_*_dev_*/*/timing.json")):
    d = json.load(open(p))
    if d.get("gen_s") is not None and "chunk_s" in d: continue
    src = os.path.join(d.get("source", ""), "timing.json")
    if not os.path.exists(src): miss.append(p); continue
    J = json.load(open(src)); ch = sorted(J["chunks"], key=lambda r: r["c"]); tc = [r["total_s"] for r in ch]
    d["chunk_s"] = tc; d["draft_s"] = [r["draft_s"] for r in ch]; d["gen_s"] = round(sum(tc), 3)
    d["fix_1020"] = "chunk_s / draft_s / gen_s re-derived from the raw device timing.json (gen/fix_dev_gen_s_1020.py)"
    if not dry: json.dump(d, open(p, "w"), indent=1)
    n[p.split("/")[-3]] += 1
print("patched" if not dry else "would patch", sum(n.values()), dict(n)); print("missing raw:", len(miss), miss[:5])
