#!/usr/bin/env python3
"""AR-pair results on the S05 seed set: Base-likeness axis (Base vs Fast LDA, fit on chunks 6-12 of Base/Fast 480p), energy distance to Base seeds (chunks 6-12), DINO to same-seed Base, plus VGGT-free jump proxy (DINO frame-to-frame max).
Arms: 480p refs (Base, Fast, ours, ideal) + 360p: fast_only(14B), pretrain_only, periodic:3/2 (Fast->pretrain), 1.3B fast_only / periodic:3/2 (1.3B->Fast14B)."""
import os, json, glob, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S="/home/hk28965/specwm"; R=S+"/lingbot-world-v2/specwm_out"; O=R+"/spec_ar"; os.makedirs(O,exist_ok=True); torch.hub.set_dir(S+"/.torch_hub")
m=torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).eval().cuda(); mean=torch.tensor([0.485,0.456,0.406]).view(3,1,1).cuda(); std=torch.tensor([0.229,0.224,0.225]).view(3,1,1).cuda()
def feats(V):
    X=torch.from_numpy(np.ascontiguousarray(V)).permute(0,3,1,2).float().cuda()/255; X=torch.nn.functional.interpolate(X,size=(518,518),mode="bilinear"); X=(X-mean)/std; o=[]
    with torch.no_grad():
        for i in range(0,len(X),16): o.append(torch.nn.functional.normalize(m(X[i:i+16]).flatten(1),dim=-1).cpu())
    return torch.cat(o).numpy()
ARMS={"base":("Base 70-step 480p (target of the Base pair)","tab:blue","-"),"fast":("Fast 14B 480p 4-step","tab:red","-"),"ours":("Fast + ours (KV-rewrite, Base target)","tab:green","-"),"ideal":("Fast + ideal (Base KV)","tab:purple","-"),
      "fast_only":("Fast 14B 360p 4-step (draft)","salmon","--"),"pretrain_only":("causal-pretrain 40-step CFG 360p (AR target)","navy","--"),"periodic_3":("Fast→pretrain periodic every 3 chunks","teal","--"),"periodic_2":("Fast→pretrain periodic every 2 chunks","darkcyan","--"),
      "p13_fast_only":("1.3B 360p 4-step (draft)","orange",":"),"p13_periodic_3":("1.3B→Fast14B periodic every 3","goldenrod",":"),"p13_periodic_2":("1.3B→Fast14B periodic every 2","darkgoldenrod",":")}
seeds=sorted({int(d.split("_s")[-1]) for d in glob.glob(R+"/floor/base/S05_s*")}); F={a:{} for a in ARMS}
for sd in seeds:
    P={"base":R+"/floor/base/S05_s%d"%sd,"fast":R+"/spec_dist/S05/W18_sink6/fast_alone_s%d"%sd,"ours":R+"/spec_dist/S05/W18_sink6/kvrw_laundered_s%d"%sd,"ideal":R+"/spec_dist/S05/W18_sink6/resync_k6_s%d"%sd,
       "fast_only":R+"/spec_ar/S05/fast_only_s%d"%sd,"pretrain_only":R+"/spec_ar/S05/pretrain_only_s%d"%sd,"periodic_3":R+"/spec_ar/S05/periodic_3_s%d"%sd,"periodic_2":R+"/spec_ar/S05/periodic_2_s%d"%sd,
       "p13_fast_only":R+"/spec_ar_1p3b/S05/fast_only_s%d"%sd,"p13_periodic_3":R+"/spec_ar_1p3b/S05/periodic_3_s%d"%sd,"p13_periodic_2":R+"/spec_ar_1p3b/S05/periodic_2_s%d"%sd}
    if not os.path.exists(P["fast"]+"/frames_all.npy"): P["fast"]=R+"/floor/fast/S05/W18_sink6/fast_alone_s%d"%sd
    for a,p in P.items():
        if os.path.exists(p+"/frames_all.npy"): F[a][sd]=feats(np.load(p+"/frames_all.npy"))[:39]
n={a:len(F[a]) for a in ARMS}; print(n,flush=True); T=39; chunks=list(range(0,T,3)); t=np.array(chunks)/4.0
XB=np.concatenate([F["base"][s][18:] for s in F["base"]]); XF=np.concatenate([F["fast"][s][18:] for s in F["fast"]]); mb,mf=XB.mean(0),XF.mean(0); Sw=np.cov(np.concatenate([XB-mb,XF-mf]).T)+1e-3*np.eye(XB.shape[1]); w=np.linalg.solve(Sw,mb-mf); w/=np.linalg.norm(w)
PR={a:{s:np.array([F[a][s][c:c+3].mean(0)@w for c in chunks]) for s in F[a]} for a in ARMS if F[a]}
def energy(X,Y):
    md=lambda A,B: np.mean(1-A@B.T); return float(2*md(X,Y)-md(X,X)-md(Y,Y))
def late_energy(a): return float(np.mean([energy(np.stack([F[a][s][c:c+3].mean(0) for s in F[a]]),np.stack([F["base"][s][c:c+3].mean(0) for s in F["base"]])) for c in chunks[6:]]))
def paired(a): ss=[s for s in F[a] if s in F["base"]]; return float(np.mean([np.mean([1-(F[a][s][c:c+3]*F["base"][s][c:c+3]).sum(-1).mean() for c in chunks[6:]]) for s in ss]))
def jmax(a): return float(np.mean([(1-(F[a][s][1:]*F[a][s][:-1]).sum(-1)).max() for s in F[a]]))
fig,ax=plt.subplots(figsize=(15,6.5))
for a in ARMS:
    if a not in PR: continue
    M=np.stack(list(PR[a].values())); med=np.median(M,0); q1,q3=np.percentile(M,25,0),np.percentile(M,75,0); lab,col,ls=ARMS[a]
    ax.plot(t,med,color=col,ls=ls,lw=2.2 if ls!="-" else 1.6,marker="o",ms=3,label="%s (n=%d)"%(lab,len(M))); ax.fill_between(t,q1,q3,color=col,alpha=.08)
ax.axvspan(4.5,4.75,color="gold",alpha=.25); ax.set_xlabel("time (s)"); ax.set_ylabel("Base-likeness (Base-vs-Fast LDA axis) →"); ax.set_title("S05 seeds — Base pair (solid, 480p) vs AR pairs (dashed = Fast→pretrain, dotted = 1.3B→Fast14B; 360p)",fontsize=11); ax.legend(fontsize=7.5,loc="lower left",ncol=2); fig.tight_layout(); fig.savefig(O+"/baselike_ar.png",dpi=110); plt.close(fig)
md="| arm | n | Base-likeness 4.5–9 s ↑ | energy dist. to Base seeds (6–12) ↓ | DINO to same-seed Base (6–12) ↓ | frame-to-frame jump max ↓ |\n|---|---|---|---|---|---|\n"
for a in ARMS:
    if a not in PR: continue
    M=np.stack(list(PR[a].values())); md+="| %s | %d | %+.3f | %.3f | %.3f | %.3f |\n"%(ARMS[a][0],len(M),np.median(M,0)[6:].mean(),late_energy(a) if a!="base" else 0.0,paired(a) if a!="base" else 0.0,jmax(a))
open(O+"/summary_ar.md","w").write(md); print(md)
