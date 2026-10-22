#!/usr/bin/env python3
"""summarize probe2 runs: per preset mean check time, Spearman corr with full across chunks, top-2 chunks by distance. usage: probe2_summary.py"""
import json, glob, os, numpy as np
from scipy.stats import spearmanr
R="/home/hk28965/specwm/lingbot-world-v2/specwm_out/spec_v0"
for p in sorted(glob.glob(R+"/*/probe2_s*/spec_v0.json")):
    if "_smoke" in p: continue
    j=json.load(open(p)); t=p.split("/")[-3]; sd=j["seed"]; D={}
    for l in j["log"]:
        for r in (l.get("nodes") or []): D.setdefault(r["preset"],[]).append((l["chunk"],r["rel"],r["t_cond"]+r["t_fwd"],r["t_fwd"]))
    full=np.array([x[1] for x in D["full"]]); print("== %s s%d"%(t,sd))
    for k,v in D.items():
        rel=np.array([x[1] for x in v]); ch=[x[0] for x in v]; top=[ch[i] for i in np.argsort(-rel)[:2]]
        print("  %-9s fwd %5.2fs  total %5.2fs  rho %.2f  top chunks %s"%(k,np.mean([x[3] for x in v]),np.mean([x[2] for x in v]),spearmanr(rel,full)[0],top))
