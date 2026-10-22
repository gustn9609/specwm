#!/usr/bin/env python3
"""Verification V-a/V-e: on all 85 StEvo camera tasks (seed 1234, 161 f), DINO chunks 6-12 distance to Base70 (253 f, first 161 used) for fast_alone / kvrw_laundered (ours) / resync_k6 (ideal),
plus frame-to-frame jump statistics (max, #spikes>0.3) and lock-on to provider. Paired stats: mean Δ, 95% CI (t), sign test, recovery."""
import os, json, glob, numpy as np, torch
from scipy import stats
S="/home/hk28965/specwm"; R=S+"/lingbot-world-v2/specwm_out"; RG=R+"/stevo/reground"; torch.hub.set_dir(S+"/.torch_hub")
m=torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).eval().cuda()
mean=torch.tensor([0.485,0.456,0.406]).view(3,1,1).cuda(); std=torch.tensor([0.229,0.224,0.225]).view(3,1,1).cuda()
def feats(V):
    X=torch.from_numpy(np.ascontiguousarray(V)).permute(0,3,1,2).float().cuda()/255; X=torch.nn.functional.interpolate(X,size=(518,518),mode="bilinear"); X=(X-mean)/std; o=[]
    with torch.no_grad():
        for i in range(0,len(X),16): o.append(torch.nn.functional.normalize(m(X[i:i+16]).flatten(1),dim=-1).cpu())
    return torch.cat(o)
def dist(A,B): n=min(len(A),len(B)); return (1-(A[:n]*B[:n]).sum(-1)).numpy()
def pc(d): return [float(d[c*3:(c+1)*3].mean()) for c in range(len(d)//3)]
def jumps(F): j=1-(F[1:]*F[:-1]).sum(-1); j=j.numpy(); return float(j.max()), int((j>0.3).sum())
rows={}
for td in sorted(glob.glob(RG+"/*/W18_sink6")):
    t=td.split("/")[-2]; bp=R+"/stevo/lingbot_base/%s/frames_all.npy"%t
    if not os.path.exists(bp): continue
    FB=feats(np.load(bp)[:41]); r={}
    for arm in ["fast_alone","kvrw_laundered","resync_k6"]:
        p=td+"/%s_s1234/frames_all.npy"%arm
        if not os.path.exists(p): continue
        F=feats(np.load(p)); d=pc(dist(F,FB)); jm,ns=jumps(F[6*3:]); r[arm]=dict(d612=float(np.mean(d[6:])), d_all=float(np.mean(d)), jump_max=jm, spikes=ns, chunks=d)
    if len(r)==3: rows[t]=r; print(t, {a:round(v["d612"],3) for a,v in r.items()}, flush=True)
json.dump(rows,open(R+"/verify_ours_stevo.json","w"),indent=1)
n=len(rows); fa=np.array([rows[t]["fast_alone"]["d612"] for t in rows]); ou=np.array([rows[t]["kvrw_laundered"]["d612"] for t in rows]); idl=np.array([rows[t]["resync_k6"]["d612"] for t in rows])
def paired(x,y,lab):
    d=x-y; ci=stats.t.interval(0.95,len(d)-1,loc=d.mean(),scale=stats.sem(d)); w=(d>0).sum(); l=(d<0).sum(); p=stats.binomtest(int(w),int(w+l)).pvalue; tt=stats.ttest_rel(x,y).pvalue
    return "| %s | %.3f ± %.3f | [%.3f, %.3f] | %d / %d | %.3g | %.3g |"%(lab,d.mean(),d.std(),ci[0],ci[1],w,l,p,tt)
md="# Verification: Fast + ours on all StEvo camera tasks (n=%d, seed 1234, 161 f; DINO chunks 6-12 to Base70 ↓)\n\n| arm | DINO 6-12 ↓ (mean ± std) | frame-to-frame jump max ↓ | #spikes>0.3 per clip ↓ |\n|---|---|---|---|\n"%n
for arm,lab in [("fast_alone","Fast alone"),("kvrw_laundered","Fast + ours (KV-rewrite)"),("resync_k6","Fast + ideal (Base ctx)")]:
    x=np.array([rows[t][arm]["d612"] for t in rows]); jm=np.array([rows[t][arm]["jump_max"] for t in rows]); ns=np.array([rows[t][arm]["spikes"] for t in rows])
    md+="| %s | %.3f ± %.3f | %.3f | %.2f |\n"%(lab,x.mean(),x.std(),jm.mean(),ns.mean())
md+="\n| paired comparison | ΔDINO (mean ± std) | 95%% CI | wins / losses | sign-test p | paired t p |\n|---|---|---|---|---|---|\n"+paired(fa,ou,"Fast alone − ours (positive = ours closer to Base)")+"\n"+paired(fa,idl,"Fast alone − ideal")+"\n"+paired(ou,idl,"ours − ideal")+"\n"
rec=(fa-ou)/(fa-idl); ok=(fa-idl)>0.02; md+="\nrecovery = (fast−ours)/(fast−ideal): mean %.0f%% ± %.0f%% (n=%d combos with fast−ideal>0.02); median %.0f%%; pooled (Σ(fast−ours)/Σ(fast−ideal)) = %.0f%%\n"%(100*rec[ok].mean(),100*rec[ok].std(),ok.sum(),100*np.median(rec[ok]),100*(fa-ou).sum()/(fa-idl).sum())
open(R+"/verify_ours_stevo.md","w").write(md); print(md)
