"""Action-edit refiner (item 2) harness report: the edit arm (K1, misses edited by one LoRA forward) against K1 regenerate (same hits) and K0.
usage: envrun python scripts/edit_report.py ROOT EDIT_ARM [BASE_ARM=uniform4_fx] [--vlm ROOT/vlm.json]"""
import sys, os, glob, json, re, numpy as np
root = sys.argv[1]; ed = sys.argv[2]; base = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else "uniform4_fx"; VLM = json.load(open(sys.argv[sys.argv.index("--vlm") + 1])) if "--vlm" in sys.argv else json.load(open(root + "/vlm.json"))
def load(arm, K):
    R = {}
    for p in glob.glob(root + "/*/*/K%d_%s_s*/loop.json" % (K, arm)):
        if "/_warm/" in p or not re.match(r"^K%d_%s_s\d+$" % (K, re.escape(arm)), os.path.basename(os.path.dirname(p))): continue
        L = json.load(open(p)); L["_dir"] = os.path.dirname(p); R[(L["task"], L["stream"])] = L
    return R
E = load(ed, 1); B1 = load(base, 1); B0 = load(base, 0); fam = lambda s: re.sub(r"_s\d+$", "", s)
def vq(L): v = VLM.get(L["_dir"] + "/video.mp4"); return (v["quality"], v["drift"]) if v else (None, None)
def lat(L, which=None):
    cs = L["chunk_start"]; rows = {r["c"]: r for r in L["rows"]}; out = []
    for _, c, t in L["frame_ready"]:
        if c >= 1 and (which is None or (which == "hit") == (rows[c]["hit"] is not None)): out.append(1000 * (t - cs[c]))
    return out
print("| family | clips | hit rate | edited chunks: within-ε vs true-action chunk (as-is branch) | miss-chunk ms: edit / regenerate ↓ | mean ms: edit / regen / K0 ↓ | fps: edit / regen / K0 ↑ | quality: edit / regen / K0 ↑ | Δq edit−regen (paired) | Δq edit−K0 | drift: edit / regen / K0 ↓ |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for f in sorted(set(fam(s) for _, s in E)):
    keys = sorted(k for k in E if fam(k[1]) == f and k in B1 and k in B0)
    if not keys: continue
    wE = [r["oracle"]["within"] for k in keys for r in E[k]["rows"] if "edit" in r and "oracle" in r]; wA = [r["oracle"]["within_asis"] for k in keys for r in E[k]["rows"] if "edit" in r and "oracle" in r]
    mE = [x for k in keys for x in lat(E[k], "miss")]; mB = [x for k in keys for x in lat(B1[k], "miss")]
    aE = [x for k in keys for x in lat(E[k])]; aB = [x for k in keys for x in lat(B1[k])]; a0 = [x for k in keys for x in lat(B0[k])]
    hr = sum(E[k]["hits"] for k in keys) / max(sum(E[k]["spec_chunks"] for k in keys), 1)
    q = [(vq(E[k]), vq(B1[k]), vq(B0[k])) for k in keys]; q = [x for x in q if all(v[0] is not None for v in x)]
    qe, qb, q0 = (np.array([x[i][0] for x in q]) for i in range(3)); de, db, d0 = (np.array([x[i][1] for x in q]) for i in range(3))
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    print("| %s | %d | %.0f %% | %.0f %% (%.0f %%), n=%d | %.0f / %.0f | %.0f / %.0f / %.0f | %.2f / %.2f / %.2f | %.2f / %.2f / %.2f | %+.2f ± %.2f | %+.2f ± %.2f | %.1f / %.1f / %.1f |" % (
        f, len(keys), 100 * hr, 100 * np.mean(wE), 100 * np.mean(wA), len(wE), np.mean(mE), np.mean(mB), np.mean(aE), np.mean(aB), np.mean(a0),
        np.mean([E[k]["fps"] for k in keys]), np.mean([B1[k]["fps"] for k in keys]), np.mean([B0[k]["fps"] for k in keys]), qe.mean(), qb.mean(), q0.mean(), (qe - qb).mean(), se(qe - qb), (qe - q0).mean(), se(qe - q0), de.mean(), db.mean(), d0.mean()))
