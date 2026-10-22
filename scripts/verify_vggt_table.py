import json,glob,os,numpy as np
from scipy import stats
B="lingbot-world-v2/specwm_out/verify_vggt"; d={}
for p in glob.glob(B+"/*.json"):
    t,a=os.path.basename(p)[:-5].split("__"); d.setdefault(t,{})[a]=json.load(open(p))
tasks=[t for t in d if all(a in d[t] for a in ("fast_alone","ours","ideal","base"))]; out="n=%d\n\n"%len(tasks)
out+="| arm | VGGT pos err vs commanded (rel) ↓ | fwd-angle err (deg) ↓ | rot-jump max (deg) ↓ | rot-jump p95 ↓ |\n|---|---|---|---|---|\n"; M={}
for a,lab in [("fast_alone","Fast alone"),("ours","Fast + ours"),("ideal","Fast + ideal"),("base","Base 70-step (first 10 s)")]:
    pe=np.array([d[t][a]["mean_pos_err_rel"] for t in tasks]); fe=np.array([d[t][a]["mean_fwd_err_deg"] for t in tasks]); rj=np.array([max(d[t][a]["rot_jump_deg"]) for t in tasks]); p95=np.array([np.percentile(d[t][a]["rot_jump_deg"],95) for t in tasks]); M[a]=(pe,fe,rj,p95)
    out+="| %s | %.3f | %.1f | %.1f | %.1f |\n"%(lab,pe.mean(),fe.mean(),rj.mean(),p95.mean())
for ref,lab in [("ours","Fast alone − ours"),("ideal","Fast alone − ideal")]:
    out+="\n| paired (%s) | mean Δ (↑ = better) | wins/losses | sign p | t p |\n|---|---|---|---|---|\n"%lab
    for i,n in enumerate(["pos err","fwd err","rot-jump max","rot-jump p95"]):
        x=M["fast_alone"][i]; y=M[ref][i]; dd=x-y; w=int((dd>0).sum()); l=int((dd<0).sum()); out+="| %s | %+.3f | %d / %d | %.3f | %.3f |\n"%(n,dd.mean(),w,l,stats.binomtest(w,w+l).pvalue,stats.ttest_rel(x,y).pvalue)
open("lingbot-world-v2/specwm_out/verify_vggt.md","w").write(out); print(out)
