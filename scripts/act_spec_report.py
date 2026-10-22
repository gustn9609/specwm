"""Aggregate specwm/act_spec.py runs (<root>/<task>/<stream>/K<k>_<arm>_s<seed>/loop.json): per stream family x K: hit rate, action->frame
latency (all frames, p50 / mean; hit vs miss chunks), output fps, generations per chunk (branches + regenerations), oracle within-eps on hits,
and the quality / drift of the videos if a VLM json is given (paired with the K0 run of the same task x stream).
usage: python scripts/act_spec_report.py ROOT [--arm uniform4] [--vlm vlm.json] [--json OUT.json] [--md OUT.md]"""
import sys, os, glob, json, re, numpy as np
root = sys.argv[1]; arm = sys.argv[sys.argv.index("--arm") + 1] if "--arm" in sys.argv else "uniform4"; VLM = json.load(open(sys.argv[sys.argv.index("--vlm") + 1])) if "--vlm" in sys.argv else {}
runs = {}
for p in sorted(glob.glob(root + "/*/*/K*_%s_s*/loop.json" % arm)):
    if "/_warm/" in p: continue
    L = json.load(open(p)); runs[(L["task"], L["stream"], L["K"])] = L; L["_dir"] = os.path.dirname(p)
fam = lambda s: "real" if s.startswith("real_") else ("vpt" if s.startswith("vpt_") else re.sub(r"_s\d+$", "", s)); families = sorted(set(fam(s) for _, s, _ in runs), key=lambda f: (f.startswith("tau2j"), f)); Ks = sorted(set(k for _, _, k in runs))
def vq(L):
    v = VLM.get(L["_dir"] + "/video.mp4"); return (v["quality"], v["drift"]) if v else (None, None)
rows = []; out = {}
for f in families:
    for K in Ks:
        R = [L for (t, s, k), L in runs.items() if fam(s) == f and k == K]
        if not R: continue
        lat = []; lat_hit = []; lat_miss = []; hits = 0; spec = 0; gens = 0; chunks = 0; orc = []; fps = []; dq = []; dd = []; q = []; d = []; steps = []; f1 = 0; lf = 0
        for L in R:
            cs = L["chunk_start"]; rows_ = {r["c"]: r for r in L["rows"]}
            for _, c, t in L["frame_ready"]:
                ms = 1000 * (t - cs[c]); lat.append(ms)
                if c >= 1 and K >= 1: (lat_hit if rows_[c]["hit"] is not None else lat_miss).append(ms)
            hits += L["hits"]; spec += L["spec_chunks"]; chunks += L["chunks"]; fps.append(L["fps"]); steps.append(L["mean_steps"]); f1 += L.get("frames_1", 0); lf += L.get("lat_frames", 3 * L["chunks"])
            gens += sum(len(r.get("branches", [])) + (0 if r["hit"] is not None else 1) for r in L["rows"])
            orc += [r["oracle"]["within"] for r in L["rows"] if "oracle" in r]
            qq, dd_ = vq(L); b = runs.get((L["task"], L["stream"], 0))
            if qq is not None: q.append(qq); d.append(dd_)
            if qq is not None and b is not None and K > 0:
                bq, bd = vq(b)
                if bq is not None: dq.append(qq - bq); dd.append(dd_ - bd)
        r = dict(family=f, K=K, clips=len(R), hit_rate=hits / max(spec, 1), hits=hits, spec_chunks=spec, p50=float(np.percentile(lat, 50)), mean=float(np.mean(lat)), p90=float(np.percentile(lat, 90)),
                 hit_ms=float(np.mean(lat_hit)) if lat_hit else None, frames1_share=f1 / max(lf, 1), miss_ms=float(np.mean(lat_miss)) if lat_miss else None, fps=float(np.mean(fps)), gens_per_chunk=gens / chunks, oracle_within=float(np.mean(orc)) if orc else None,
                 q=float(np.mean(q)) if q else None, d=float(np.mean(d)) if d else None, dq=float(np.mean(dq)) if dq else None, dq_se=float(np.std(dq) / np.sqrt(len(dq))) if len(dq) > 1 else None, dd=float(np.mean(dd)) if dd else None, n_q=len(q))
        rows.append(r); out["%s_K%d" % (f, K)] = r
fmt = lambda v, f="%.0f": "–" if v is None else (f % v)
print("| stream family | K | clips | hit rate ↑ | action→frame p50 ms ↓ | mean ↓ | hit chunks ms | miss chunks ms | output fps ↑ | generations / chunk ↓ | oracle within ε on hits ↑ | 1-frame units (share of frames) | quality ↑ (Δ vs K0) | drift ↓ (Δ) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print("| %s | %d | %d | %s | %s | %s | %s | %s | %.1f | %.2f | %s | %s | %s (%s) | %s (%s) |" % (r["family"], r["K"], r["clips"], fmt(100 * r["hit_rate"], "%.0f %%") if r["K"] else "–", fmt(r["p50"]), fmt(r["mean"]), fmt(r["hit_ms"]), fmt(r["miss_ms"]), r["fps"], r["gens_per_chunk"],
          fmt(100 * r["oracle_within"], "%.0f %%") if r["oracle_within"] is not None else "–", fmt(100 * r["frames1_share"], "%.0f %%"), fmt(r["q"], "%.2f"), fmt(r["dq"], "%+.2f") + (" ± %.2f" % r["dq_se"] if r["dq_se"] else ""), fmt(r["d"], "%.1f"), fmt(r["dd"], "%+.1f")))
if "--json" in sys.argv: json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
