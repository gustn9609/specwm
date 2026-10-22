#!/usr/bin/env python3
"""Clearer views of the seed-distribution experiment (S05): (A) per-time 2D scatter + 2σ ellipses, (B) 1-D 'Base-likeness' axis (LDA Base vs Fast on late chunks) as violins over time,
(C) mean±std trajectory bands, (D) frame grid at 6 s (seeds x arms)."""
import os, json, glob, numpy as np, torch, imageio, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib.patches import Ellipse
S="/home/hk28965/specwm"; R=S+"/lingbot-world-v2/specwm_out"; O=R+"/spec_dist/S05"; torch.hub.set_dir(S+"/.torch_hub")
m=torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).eval().cuda(); mean=torch.tensor([0.485,0.456,0.406]).view(3,1,1).cuda(); std=torch.tensor([0.229,0.224,0.225]).view(3,1,1).cuda()
def feats(V):
    X=torch.from_numpy(np.ascontiguousarray(V)).permute(0,3,1,2).float().cuda()/255; X=torch.nn.functional.interpolate(X,size=(518,518),mode="bilinear"); X=(X-mean)/std; o=[]
    with torch.no_grad():
        for i in range(0,len(X),16): o.append(torch.nn.functional.normalize(m(X[i:i+16]).flatten(1),dim=-1).cpu())
    return torch.cat(o).numpy()
ARMS={"base":("Base 70-step (target)","tab:blue"),"fast":("Fast 4-step (draft)","tab:red"),"ours":("Fast + ours (KV-rewrite)","tab:green"),"ideal":("Fast + ideal (Base KV)","tab:purple")}
seeds=sorted({int(d.split("_s")[-1]) for d in glob.glob(R+"/floor/base/S05_s*")}); F={a:{} for a in ARMS}; VID={a:{} for a in ARMS}
for sd in seeds:
    P={"base":R+"/floor/base/S05_s%d"%sd,"fast":R+"/spec_dist/S05/W18_sink6/fast_alone_s%d"%sd,"ours":R+"/spec_dist/S05/W18_sink6/kvrw_laundered_s%d"%sd,"ideal":R+"/spec_dist/S05/W18_sink6/resync_k6_s%d"%sd}
    if not os.path.exists(P["fast"]+"/frames_all.npy"): P["fast"]=R+"/floor/fast/S05/W18_sink6/fast_alone_s%d"%sd
    for a,p in P.items():
        if os.path.exists(p+"/frames_all.npy"): V=np.load(p+"/frames_all.npy")[:39]; F[a][sd]=feats(V); VID[a][sd]=V
T=39; times=[9,18,24,36]; tlab={9:"2.25 s",18:"4.5 s (switch)",24:"6 s",36:"9 s"}
# ---- (A) per-time scatter with 2-sigma ellipses; PCA fit per time slice on all arms
fig,axes=plt.subplots(1,len(times),figsize=(5.2*len(times),5))
for ax,ti in zip(axes,times):
    X=np.concatenate([np.stack([F[a][s][ti] for s in F[a]]) for a in ARMS]); mu=X.mean(0); U,Sv,Vt=np.linalg.svd(X-mu,full_matrices=False)
    for a in ARMS:
        P2=(np.stack([F[a][s][ti] for s in F[a]])-mu)@Vt[:2].T; ax.scatter(P2[:,0],P2[:,1],s=28,color=ARMS[a][1],alpha=.75,label="%s (n=%d)"%(ARMS[a][0],len(P2)),edgecolor="k",linewidths=.3)
        c=np.cov(P2.T); w,v=np.linalg.eigh(c); ang=np.degrees(np.arctan2(v[1,1],v[0,1])); ax.add_patch(Ellipse(P2.mean(0),2*2*np.sqrt(w[1]),2*2*np.sqrt(w[0]),angle=ang,fill=False,color=ARMS[a][1],lw=2))
    ax.set_title("t = %s — one point per seed (2σ ellipses)"%tlab[ti],fontsize=10); ax.set_xticks([]); ax.set_yticks([])
axes[0].legend(fontsize=8,loc="lower left"); fig.suptitle("S05: where the seeds land at a given time (PCA of that time slice)",fontsize=12); fig.tight_layout(); fig.savefig(O+"/slices.png",dpi=110); plt.close(fig)
# ---- (B) Base-likeness axis: LDA direction Base vs Fast fit on chunks 6-12, projected per chunk
XB=np.concatenate([F["base"][s][18:] for s in F["base"]]); XF=np.concatenate([F["fast"][s][18:] for s in F["fast"]]); mb,mf=XB.mean(0),XF.mean(0); Sw=np.cov(np.concatenate([XB-mb,XF-mf]).T)+1e-3*np.eye(XB.shape[1]); w=np.linalg.solve(Sw,mb-mf); w/=np.linalg.norm(w)
chunks=list(range(0,T,3)); t=np.array(chunks)/4.0
fig,ax=plt.subplots(figsize=(14,5)); off={"base":-0.3,"fast":-0.1,"ours":0.1,"ideal":0.3}
for a in ARMS:
    data=[np.array([F[a][s][c:c+3].mean(0)@w for s in F[a]]) for c in chunks]; med=[np.median(d) for d in data]; q1=[np.percentile(d,25) for d in data]; q3=[np.percentile(d,75) for d in data]
    ax.plot(t,med,color=ARMS[a][1],lw=2,marker="o",label=ARMS[a][0]); ax.fill_between(t,q1,q3,color=ARMS[a][1],alpha=.15)
    for tt,d in zip(t,data): ax.scatter(np.full(len(d),tt+off[a]*0.25),d,s=6,color=ARMS[a][1],alpha=.5)
ax.axvspan(4.5,4.75,color="gold",alpha=.3); ax.text(4.6,ax.get_ylim()[1],"KV switch",ha="center",va="top",fontsize=8,color="darkgoldenrod")
ax.set_xlabel("time (s)"); ax.set_ylabel("projection on Base-vs-Fast axis  (→ more Base-like)"); ax.set_title("1-D 'Base-likeness' per chunk: median line, IQR band, dots = seeds",fontsize=11); ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(O+"/baselike_axis.png",dpi=110); plt.close(fig)
# ---- (C) mean ± std trajectory in a shared PCA of chunk means
allc=np.concatenate([np.stack([F[a][s][c:c+3].mean(0) for c in chunks]) for a in ARMS for s in F[a]]); mu=allc.mean(0); U,Sv,Vt=np.linalg.svd(allc-mu,full_matrices=False)
fig,ax=plt.subplots(figsize=(8,7))
for a in ARMS:
    P=np.stack([(np.stack([F[a][s][c:c+3].mean(0) for c in chunks])-mu)@Vt[:2].T for s in F[a]]); M=P.mean(0); Sd=P.std(0)
    ax.plot(M[:,0],M[:,1],color=ARMS[a][1],lw=2.5,marker="o",ms=4,label=ARMS[a][0])
    for k in range(len(chunks)): ax.add_patch(Ellipse(M[k],2*Sd[k,0],2*Sd[k,1],fill=True,color=ARMS[a][1],alpha=.06))
    ax.annotate("0 s",M[0],fontsize=8,color=ARMS[a][1]); ax.annotate("9 s",M[-1],fontsize=8,color=ARMS[a][1]); ax.scatter(M[6,0],M[6,1],s=90,facecolor="none",edgecolor="gold",linewidths=2)
ax.set_title("mean trajectory per arm (chunk means, shared PCA); shaded = ±1σ across seeds; gold ring = 4.5 s switch",fontsize=10); ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(O+"/mean_traj.png",dpi=110); plt.close(fig)
# ---- (D) frame grid at 6 s: 8 seeds x 4 arms
ti=24; ss=[s for s in seeds if all(s in F[a] for a in ARMS)][:8]; tw,th=200,112; rows=[]
for a in ARMS:
    tiles=[cv2.resize(VID[a][s][ti],(tw,th)) for s in ss]; strip=np.concatenate(tiles,1); lbl=np.full((22,strip.shape[1],3),245,np.uint8); cv2.putText(lbl,ARMS[a][0],(6,16),cv2.FONT_HERSHEY_SIMPLEX,0.55,(30,30,30),1); rows+=[lbl,strip]
grid=np.concatenate(rows,0); hdr=np.full((22,grid.shape[1],3),255,np.uint8)
for i,s in enumerate(ss): cv2.putText(hdr,"seed %d"%s,(i*tw+6,16),cv2.FONT_HERSHEY_SIMPLEX,0.5,(80,80,80),1)
imageio.imwrite(O+"/frames_6s.jpg",np.concatenate([hdr,grid],0),quality=85); print("[VIZ2_DONE]")
