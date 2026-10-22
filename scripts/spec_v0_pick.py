#!/usr/bin/env python3
"""helpers for the night orchestrator. usage:
  spec_v0_pick.py corr   -> prints the cheapest sweep variant's args whose mean DINO(3-10 s) over both tasks <= every:2 (16-step, full ctx) + 0.03 (else '')
  spec_v0_pick.py thr <task> <seed> <node> <q>  -> prints the q-quantile of the probe rel. distance at that node (chunks >= 1)"""
import sys, json, glob, os, numpy as np
R="/home/hk28965/specwm/lingbot-world-v2/specwm_out/spec_v0"
VAR={"steps4":"--steps 4","ctx9":"--ctx 9","steps1":"--steps 1","guide1":"--guide 1","steps8":"--steps 8","360p":"--max-area 184320","360p_steps4_ctx9":"--max-area 184320 --steps 4 --ctx 9","360p_steps4":"--max-area 184320 --steps 4","steps4_guide1":"--steps 4 --guide 1","360p_steps4_guide1":"--max-area 184320 --steps 4 --guide 1"}
if sys.argv[1]=="corr":
    S=json.load(open(R+"/eval/summary.json")); T=list(S)
    def late(arm): v=[S[t][k]["late"] for t in T for k in S[t] if S[t][k]["arm"]==arm and S[t][k]["seed"]==1234]; return np.mean(v) if len(v)==len(T) else np.nan
    def cost(arm): v=[S[t][k].get("base_s",np.nan) for t in T for k in S[t] if S[t][k]["arm"]==arm and S[t][k]["seed"]==1234]; return np.mean(v)
    ref=late("every_2"); ok=[(cost("every_2_"+v),v) for v in VAR if late("every_2_"+v)<=ref+0.03]
    print(VAR[min(ok)[1]] if ok else "")
elif sys.argv[1]=="thr":
    _,task,seed,node,q=sys.argv[1:6]; j=json.load(open(R+"/%s/probe_s%s/spec_v0.json"%(task,seed)))
    rel=[r["rel"] for l in j["log"] if l.get("nodes") for r in l["nodes"] if r["node"]==int(node)]; print("%.3f"%np.quantile(rel,float(q)))
# probe2 thresholds: spec_v0_pick.py thr2 <task> <seed> <preset> <q>  -> q-quantile of that preset's check distance (probe2 run of the same seed)
if sys.argv[1]=="thr2":
    _,_,task,seed,pre,q=sys.argv[:6]; pd=sys.argv[6] if len(sys.argv)>6 else "probe2"; j=json.load(open(R+"/%s/%s_s%s/spec_v0.json"%(task,pd,seed)))
    rel=[r["rel"] for l in j["log"] if l.get("nodes") for r in l["nodes"] if r["preset"]==pre]; print("%.4f"%np.quantile(rel,float(q)))
# beta for the mc policy: spec_v0_pick.py beta <task> <seed> <probe dir>  -> ln2 / median(dL) (median chunk accepted with prob 0.5)
if sys.argv[1]=="beta":
    _,_,task,seed,pd=sys.argv[:5]; j=json.load(open(R+"/%s/%s_s%s/spec_v0.json"%(task,pd,seed)))
    dl=[l["verify_dist"] for l in j["log"] if l.get("verify_dist") is not None]; m=float(np.median(dl)); print("%.4f"%(np.log(2)/m if m>0 else 0.0))
