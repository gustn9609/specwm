"""253-frame KV-window test: VLM quality / drift per arm, paired against the 4-step teacher and against window 18"""
import json, os, numpy as np
O = "/home/hk28965/specwm/lingbot-world-v2/specwm_out/fit24"; V = json.load(open(O + "/vlm_s253.json"))
ARMS = [("teacher 4-step, bf16 KV, window 18", "s253_a704w18_teacher"), ("2-step fused+pc1, FP8 KV, window 12", "s253_a704w12_fp8_fp8kv"), ("2-step fused+pc1, FP8 KV, window 18", "s253_a704w18_fp8_fp8kv"), ("2-step fused+pc1, FP8 KV, window 36", "s253_a704w36_fp8_fp8kv")]
def tab(arm):
    d = {}
    for p, r in V.items():
        if "/%s/" % arm in p: d[p.split("/")[-2]] = r
    return d
T = {nm: tab(a) for nm, a in ARMS}; ref = T[ARMS[0][0]]; w18 = T[ARMS[2][0]]
def fit(a):
    try: f = json.load(open("%s/%s/fit.json" % (O, a))); return f.get("fps", float("nan")), f.get("peak_reserved_GiB", float("nan")), f.get("kv_GiB", float("nan"))
    except Exception: return float("nan"), float("nan"), float("nan")
print("| arm (704x384, 253 frames = 21 chunks, 43 StEvo prompts, seed 1234) | n | quality ↑ | drift ↓ | Δq vs teacher (paired) | Δdrift vs teacher | Δq vs window 18 | steady fps ↑ | KV GiB ↓ | peak reserved GiB ↓ |")
print("|---|---|---|---|---|---|---|---|---|---|")
for nm, a in ARMS:
    d = T[nm]; ks = sorted(d); q = np.array([d[k]["quality"] for k in ks]); dr = np.array([d[k]["drift"] for k in ks])
    c = [k for k in ks if k in ref]; dq = np.array([d[k]["quality"] - ref[k]["quality"] for k in c]); dd = np.array([d[k]["drift"] - ref[k]["drift"] for k in c])
    c2 = [k for k in ks if k in w18]; dq2 = np.array([d[k]["quality"] - w18[k]["quality"] for k in c2])
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    fps, pk, kv = fit(a)
    print("| %s | %d | %.2f | %.1f | %+.2f ± %.2f | %+.1f ± %.1f | %+.2f ± %.2f | %.1f | %.1f | %.1f |" % (nm, len(ks), q.mean(), dr.mean(), dq.mean(), se(dq), dd.mean(), se(dd), dq2.mean(), se(dq2), fps, kv, pk))
