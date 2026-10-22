"""Test-time scaling of the 14B causal-fast (direction 1 de-risk): quality / drift per step count, paired against the 4-step teacher"""
import json, os, numpy as np
O = "/home/hk28965/specwm/lingbot-world-v2/specwm_out/tts"; V = json.load(open(O + "/vlm_tts.json"))
ARMS = [("2 steps (0,500)", "s2"), ("3 steps (0,333,667)", "s3"), ("4 steps (0,250,500,750) = teacher", "s4"), ("6 steps", "s6"), ("8 steps", "s8")]
def tab(arm):
    d = {}
    for p, r in V.items():
        if "/%s/" % arm in p: d[p.split("/")[-2]] = r
    return d
T = {a: tab(a) for _, a in ARMS}; ref = T["s4"]
print("| steps (14B causal-fast, 832x480, 161 frames, 43 prompts, seed 1234, FP8 weights, bf16 KV window 18, clean commit) | n | quality ↑ | drift ↓ | Δq vs 4-step (paired) | Δdrift vs 4-step | steady fps ↑ |")
print("|---|---|---|---|---|---|---|")
for nm, arm in ARMS:
    d = T[arm]; ks = sorted(d); q = np.array([d[k]["quality"] for k in ks]); dr = np.array([d[k]["drift"] for k in ks]); c = [k for k in ks if k in ref]
    dq = np.array([d[k]["quality"] - ref[k]["quality"] for k in c]); dd = np.array([d[k]["drift"] - ref[k]["drift"] for k in c]); se = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    try: fps = json.load(open("%s/%s/fit.json" % (O, arm)))["fps"]
    except Exception: fps = float("nan")
    print("| %s | %d | %.2f | %.1f | %+.2f ± %.2f | %+.1f ± %.1f | %.1f |" % (nm, len(ks), q.mean(), dr.mean(), dq.mean(), se(dq), dd.mean(), se(dd), fps))
