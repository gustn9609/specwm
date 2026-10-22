#!/usr/bin/env python3
"""Offline verifier benchmark on the jar + sandcastle videos: per-chunk drift labels (vbench/labels.txt) vs candidate signals.
Signals: DINO semantic drift (vs start frame, vs Base frame bank NN, vs previous chunk), colour/saturation shift, Base-vs-Fast
discriminator on DINO features (leave-one-seed-out), and latent check distances from probe logs (Fast-only videos).
Out: vbench/signals.json, vbench/auc.md"""
import json, os, numpy as np, torch, imageio, cv2
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
S="/home/hk28965/specwm"; R=S+"/lingbot-world-v2/specwm_out/spec_v0"; O=R+"/vbench"; torch.hub.set_dir(S+"/.torch_hub")
vids=json.load(open(O+"/videos.json")); lab={}
for l in open(O+"/labels.txt"):
    if l.startswith("#") or not l.strip(): continue
    i,s=l.split(); lab[int(i)]=[int(ch) for ch in s]
m=torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).eval().cuda(); mean=torch.tensor([0.485,0.456,0.406]).view(3,1,1).cuda(); std=torch.tensor([0.229,0.224,0.225]).view(3,1,1).cuda()
@torch.no_grad()
def feats(V):
    out=[]
    for i in range(0,len(V),16):
        x=torch.from_numpy(np.stack(V[i:i+16])).cuda().permute(0,3,1,2).float()/255; x=torch.nn.functional.interpolate(x,size=(224,392),mode="bilinear",antialias=True); x=(x-mean)/std
        out.append(torch.nn.functional.normalize(m(x),dim=-1).cpu().numpy())
    return np.concatenate(out)
def chunk_frames(n,c): return [k for k in range(max(0,12*c-2), min(n,12*c+10), 3)]
F={}; COL={}
for i,(t,sd,a,p) in enumerate(vids):
    V=[np.asarray(f) for f in imageio.get_reader(p)][:161]; idx=sorted({k for c in range(13) for k in chunk_frames(len(V),c)}|{0})
    f=feats([V[k] for k in idx]); fm={k:f[j] for j,k in enumerate(idx)}
    F[i]=np.stack([np.mean([fm[k] for k in chunk_frames(len(V),c)],0) for c in range(13)]); F[i]=F[i]/np.linalg.norm(F[i],axis=1,keepdims=True); F[(i,"f0")]=fm[0]
    hsv=[cv2.cvtColor(V[k],cv2.COLOR_RGB2HSV).reshape(-1,3).astype(np.float32) for k in idx]; hm={k:h for k,h in zip(idx,hsv)}
    COL[i]=[(np.mean([hm[k][:,1].mean() for k in chunk_frames(len(V),c)]), np.mean([hm[k][:,2].mean() for k in chunk_frames(len(V),c)]), np.mean([np.median(hm[k][:,0]) for k in chunk_frames(len(V),c)])) for c in range(13)]
    print(i,t[:6],sd,a,flush=True)
# Base frame bank per task (all base70 chunks), NN distance (leave the same video out)
sig={}
for i,(t,sd,a,p) in enumerate(vids):
    bank=np.concatenate([F[j] for j,(t2,sd2,a2,_) in enumerate(vids) if t2==t and a2=="base70" and j!=i])
    s0=np.array(COL[i][0])
    sig[i]=dict(
        d_first=[float(1-F[i][c]@F[(i,"f0")]) for c in range(13)],
        d_basebank=[float(1-(bank@F[i][c]).max()) for c in range(13)],
        d_prev=[0.0]+[float(1-F[i][c]@F[i][c-1]) for c in range(1,13)],
        sat_shift=[float(COL[i][c][0]-s0[0]) for c in range(13)],
        abs_colour=[float(np.abs(np.array(COL[i][c][:2])-s0[:2]).sum()) for c in range(13)])
# discriminator: Base chunks (base70 all chunks; 'C' chunks of spec runs) vs Fast chunks (never/probe2/d360_never, chunks>=1); leave-one-seed-out
def plan(i):
    t,sd,a,p=vids[i]; j=os.path.dirname(p)+"/spec_v0.json"
    return json.load(open(j))["plan"] if os.path.exists(j) else None
for sd_out in [1234,4321,9999]:
    X=[];Y=[]
    for i,(t,sd,a,p) in enumerate(vids):
        if sd==sd_out: continue
        if a=="base70": X+= [F[i][c] for c in range(1,13)]; Y+=[0]*12
        elif a in ("never","probe2","d360_never"): X+=[F[i][c] for c in range(1,13)]; Y+=[1]*12
        else:
            pl=plan(i)
            if pl: X+=[F[i][c] for c in range(1,13) if pl[c]=="C"]; Y+=[0]*sum(pl[c]=="C" for c in range(1,13))
    clf=LogisticRegression(C=1.0,max_iter=2000).fit(np.array(X),np.array(Y))
    for i,(t,sd,a,p) in enumerate(vids):
        if sd==sd_out: sig[i]["disc_pFast"]=[float(v) for v in clf.predict_proba(F[i])[:,1]]
# latent check distances (Fast-only probe videos)
for i,(t,sd,a,p) in enumerate(vids):
    for pd in ["probe2","probe2kv"]:
        j=R+"/%s/%s_s%d/spec_v0.json"%(t,pd,sd)
        if a=="probe2" and os.path.exists(j):
            for l in json.load(open(j))["log"]:
                for r in (l.get("nodes") or []): sig[i].setdefault("lat_"+r["preset"],[None]*13)[l["chunk"]]=r["rel"]
json.dump(dict(sig=sig),open(O+"/signals.json","w"))
# AUCs
def auc(keys,name,sub=None):
    ys=[];xs=[];per=[]
    for i in keys:
        if name not in sig[i] or i not in lab: continue
        y=lab[i][1:]; x=sig[i][name][1:]
        if any(v is None for v in x): continue
        ys+=y; xs+=x
        if 0<sum(y)<12: per.append(roc_auc_score(y,x))
    return (roc_auc_score(ys,xs) if len(set(ys))>1 else float("nan")), (float(np.mean(per)) if per else float("nan")), len(ys)
names=["d_basebank","disc_pFast","d_first","d_prev","sat_shift","abs_colour"]
allk=list(range(len(vids))); probe=[i for i,v in enumerate(vids) if v[2]=="probe2"]
md="# Offline verifier benchmark — jar + sandcastle, %d videos, per-chunk drift labels (chunks 1–12)\n\nAUC ↑ (0.5 = chance). pooled = all chunks; within-video = mean AUC over videos that contain both labels (robust to per-video offsets / trends).\n\n| signal | pooled AUC ↑ | within-video AUC ↑ | n chunks |\n|---|---|---|---|\n"%len(vids)
for n in names:
    a,b,k=auc(allk,n); md+="| %s | %.3f | %.3f | %d |\n"%(n,a,b,k)
md+="\n## Fast-only probe videos (%d videos) — includes the latent check distances\n\n| signal | pooled AUC ↑ | within-video AUC ↑ | n chunks |\n|---|---|---|---|\n"%len(probe)
lat=sorted({k for i in probe for k in sig[i] if k.startswith("lat_")})
for n in names+lat:
    a,b,k=auc(probe,n); md+="| %s | %.3f | %.3f | %d |\n"%(n,a,b,k)
open(O+"/auc.md","w").write(md); print(md)
