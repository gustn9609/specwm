#!/usr/bin/env python3
"""One clean figure: draft = Fast (AR), target = Base (bidirectional). Left: Base-likeness over time for 4 arms (displayed stream; ours/ideal shown from the switch).
Right: distribution distance to the Base seeds per chunk (energy distance) — same 4 arms + Base-vs-Base floor."""
import os, json, glob, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S="/home/hk28965/specwm"; R=S+"/lingbot-world-v2/specwm_out"; O=R+"/spec_dist/S05"; torch.hub.set_dir(S+"/.torch_hub")
m=torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).eval().cuda(); mean=torch.tensor([0.485,0.456,0.406]).view(3,1,1).cuda(); std=torch.tensor([0.229,0.224,0.225]).view(3,1,1).cuda()
def feats(V):
    X=torch.from_numpy(np.ascontiguousarray(V)).permute(0,3,1,2).float().cuda()/255; X=torch.nn.functional.interpolate(X,size=(518,518),mode="bilinear"); X=(X-mean)/std; o=[]
    with torch.no_grad():
        for i in range(0,len(X),16): o.append(torch.nn.functional.normalize(m(X[i:i+16]).flatten(1),dim=-1).cpu())
    return torch.cat(o).numpy()
A={"base":("Base 70-step (target)","#1f4e9c"),"fast":("Fast 4-step (draft, no correction)","#c0392b"),"ours":("Fast + KV-rewrite @4.5 s (ours ×1)","#1e8449"),"ours2":("Fast + KV-rewrite @4.5 s and @7.5 s (ours ×2)","#0b3d1e"),"ideal":("Fast + Base's own KV @4.5 s (oracle)","#7d3c98")}
seeds=sorted({int(d.split("_s")[-1]) for d in glob.glob(R+"/floor/base/S05_s*")}); F={a:{} for a in A}
for sd in seeds:
    P={"base":R+"/floor/base/S05_s%d"%sd,"fast":R+"/spec_dist/S05/W18_sink6/fast_alone_s%d"%sd,"ours":R+"/spec_dist/S05/W18_sink6/kvrw_laundered_s%d"%sd,"ours2":R+"/spec_dist/S05/W18_sink6/kvrw_second_k10_s%d"%sd,"ideal":R+"/spec_dist/S05/W18_sink6/resync_k6_s%d"%sd}
    if not os.path.exists(P["fast"]+"/frames_all.npy"): P["fast"]=R+"/floor/fast/S05/W18_sink6/fast_alone_s%d"%sd
    for a,p in P.items():
        if os.path.exists(p+"/frames_all.npy"): F[a][sd]=feats(np.load(p+"/frames_all.npy"))[:39]
T=39; chunks=list(range(0,T,3)); t=np.array(chunks)/4.0; SW=6; SW2=10
for sd in list(F["ours2"]):
    if sd in F["ours"] and sd in F["fast"]: F["ours2"][sd]=np.concatenate([F["fast"][sd][:SW*3],F["ours"][sd][SW*3:SW2*3],F["ours2"][sd][SW2*3:]])
XB=np.concatenate([F["base"][s][18:] for s in F["base"]]); XF=np.concatenate([F["fast"][s][18:] for s in F["fast"]]); mb,mf=XB.mean(0),XF.mean(0); Sw=np.cov(np.concatenate([XB-mb,XF-mf]).T)+1e-3*np.eye(XB.shape[1]); w=np.linalg.solve(Sw,mb-mf); w/=np.linalg.norm(w)
PR={a:np.stack([[F[a][s][c:c+3].mean(0)@w for c in chunks] for s in F[a]]) for a in A}
def energy(X,Y): md=lambda P,Q: np.mean(1-P@Q.T); return float(2*md(X,Y)-md(X,X)-md(Y,Y))
cm=lambda a,c: np.stack([F[a][s][c:c+3].mean(0) for s in F[a]])
bs=sorted(F["base"]); h=len(bs)//2
ED={a:[energy(cm(a,c),cm("base",c)) for c in chunks] for a in ["fast","ours","ours2","ideal"]}; ED["floor"]=[energy(np.stack([F["base"][s][c:c+3].mean(0) for s in bs[:h]]),np.stack([F["base"][s][c:c+3].mean(0) for s in bs[h:]])) for c in chunks]
fig,axes=plt.subplots(1,2,figsize=(17,5.8))
ax=axes[0]
for a in ["base","fast"]:
    M=PR[a]; ax.plot(t,np.median(M,0),color=A[a][1],lw=2.5,marker="o",label=A[a][0]); ax.fill_between(t,np.percentile(M,25,0),np.percentile(M,75,0),color=A[a][1],alpha=.15)
fm=np.median(PR["fast"],0)
for a in ["ours","ours2","ideal"]:
    M=PR[a]; med=np.median(M,0); s0=SW2 if a=="ours2" else SW; ax.plot(t[s0:],med[s0:],color=A[a][1],lw=2.5,marker="o",label=A[a][0]); ax.fill_between(t[s0:],np.percentile(M,25,0)[s0:],np.percentile(M,75,0)[s0:],color=A[a][1],alpha=.15)
    prev=np.median(PR["ours"],0) if a=="ours2" else fm; ax.plot([t[s0-1],t[s0]],[prev[s0-1],med[s0]],color=A[a][1],lw=1.2,ls="--")
ax.axvline(7.5,color="gold",lw=8,alpha=.2); ax.text(7.5,0.155,"2nd switch\n(ours ×2)",ha="center",va="top",fontsize=8,color="#8a6d00")
ax.axvline(4.5,color="gold",lw=8,alpha=.35); ax.text(4.5,0.155,"KV switch\n(visible cut)",ha="center",va="top",fontsize=9,color="#8a6d00")
ax.set_ylim(-0.21,0.16); ax.set_xlabel("time (s)",fontsize=11); ax.set_ylabel("Base-likeness  (← Fast-like    Base-like →)",fontsize=11); ax.set_title("What the viewer sees (median over 24 seeds, IQR shaded)",fontsize=11); ax.legend(fontsize=9,loc="lower left"); ax.grid(alpha=.25)
ax=axes[1]
for a,lab,col in [("fast",A["fast"][0],A["fast"][1]),("ours",A["ours"][0],A["ours"][1]),("ours2",A["ours2"][0],A["ours2"][1]),("ideal",A["ideal"][0],A["ideal"][1])]:
    e=np.array(ED[a]); s0=SW2 if a=="ours2" else SW; ax.plot(t[s0:],e[s0:],color=col,lw=2.5,marker="o",label=lab)
    if a!="ours2": ax.plot(t[:SW+1],e[:SW+1],color=col,lw=1,ls=":",alpha=.5)
ax.axvline(7.5,color="gold",lw=8,alpha=.2)
ax.plot(t,ED["floor"],color="gray",lw=2,ls="--",label="Base seeds vs Base seeds (floor)"); ax.axvline(4.5,color="gold",lw=8,alpha=.35)
ax.set_xlabel("time (s)",fontsize=11); ax.set_ylabel("energy distance to the Base seed distribution ↓",fontsize=11); ax.set_title("Distribution gap to Base per chunk (unpaired, 24 seeds each)",fontsize=11); ax.legend(fontsize=9); ax.grid(alpha=.25)
fig.suptitle("S05, one prompt, 24 seeds — draft = Fast (AR), target = Base (bidirectional)",fontsize=13); fig.tight_layout(); fig.savefig(O+"/base_pair_clean.png",dpi=120); plt.close(fig)
print("energy 7.5–9 s: fast %.3f ours×1 %.3f ours×2 %.3f ideal %.3f floor %.3f"%(np.mean(ED["fast"][SW2:]),np.mean(ED["ours"][SW2:]),np.mean(ED["ours2"][SW2:]),np.mean(ED["ideal"][SW2:]),np.mean(ED["floor"][SW2:])))
print("Base-likeness 7.5–9 s: fast %+.3f ours×1 %+.3f ours×2 %+.3f ideal %+.3f base %+.3f"%tuple(np.median(PR[a],0)[SW2:].mean() for a in ["fast","ours","ours2","ideal","base"]))
