#!/usr/bin/env python3
"""Trajectory-distribution analysis: S05, many seeds x {Base70 (target), Fast (draft), Fast+ours (speculative/KV-rewrite), ideal}.
DINOv2 features at 4 fps (frames_all.npy). Outputs: spec_dist/S05/{pca_traj.png, energy_by_chunk.png, spread.png, summary.md, summary.json}"""
import os, json, glob, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S="/home/hk28965/specwm"; R=S+"/lingbot-world-v2/specwm_out"; O=R+"/spec_dist/S05"; torch.hub.set_dir(S+"/.torch_hub")
m=torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).eval().cuda(); mean=torch.tensor([0.485,0.456,0.406]).view(3,1,1).cuda(); std=torch.tensor([0.229,0.224,0.225]).view(3,1,1).cuda()
def feats(V):
    X=torch.from_numpy(np.ascontiguousarray(V)).permute(0,3,1,2).float().cuda()/255; X=torch.nn.functional.interpolate(X,size=(518,518),mode="bilinear"); X=(X-mean)/std; o=[]
    with torch.no_grad():
        for i in range(0,len(X),16): o.append(torch.nn.functional.normalize(m(X[i:i+16]).flatten(1),dim=-1).cpu())
    return torch.cat(o).numpy()
ARMS={"base":("Base 70-step (target)","tab:blue"),"fast":("Fast 4-step (draft)","tab:red"),"ours":("Fast + ours (KV-rewrite)","tab:green"),"ideal":("Fast + ideal (Base KV)","tab:purple")}
seeds=sorted({int(d.split("_s")[-1]) for d in glob.glob(R+"/floor/base/S05_s*")})
F={a:{} for a in ARMS}
for sd in seeds:
    paths={"base":R+"/floor/base/S05_s%d/frames_all.npy"%sd,"fast":R+"/spec_dist/S05/W18_sink6/fast_alone_s%d/frames_all.npy"%sd,"ours":R+"/spec_dist/S05/W18_sink6/kvrw_laundered_s%d/frames_all.npy"%sd,"ideal":R+"/spec_dist/S05/W18_sink6/resync_k6_s%d/frames_all.npy"%sd}
    if not os.path.exists(paths["fast"]): paths["fast"]=R+"/floor/fast/S05/W18_sink6/fast_alone_s%d/frames_all.npy"%sd
    for a,p in paths.items():
        if os.path.exists(p): F[a][sd]=feats(np.load(p))[:39]
T=39; n={a:len(F[a]) for a in ARMS}; print("seeds per arm:",n,flush=True)
# ---- energy distance between arms per chunk (distribution vs distribution, seeds unpaired)
def energy(X,Y):
    def md(A,B): return np.mean(1-A@B.T)
    return float(2*md(X,Y)-md(X,X)-md(Y,Y))
chunks=list(range(0,T,3)); ED={}
for a in ["fast","ours","ideal"]:
    ED[a]=[energy(np.stack([F[a][s][c:c+3].mean(0) for s in F[a]]), np.stack([F["base"][s][c:c+3].mean(0) for s in F["base"]])) for c in chunks]
# base-vs-base floor: split base seeds in halves
bs=sorted(F["base"]); h=len(bs)//2; ED["base_split"]=[energy(np.stack([F["base"][s][c:c+3].mean(0) for s in bs[:h]]), np.stack([F["base"][s][c:c+3].mean(0) for s in bs[h:]])) for c in chunks]
# ---- spread: mean pairwise distance among seeds per chunk (does spec collapse or match Base's spread?)
SP={}
for a in ARMS:
    X=[F[a][s] for s in F[a]]; SP[a]=[float(np.mean([1-X[i][c:c+3].mean(0)@X[j][c:c+3].mean(0) for i in range(len(X)) for j in range(i+1,len(X))])) for c in chunks]
# ---- distance to the SAME-seed base per chunk (paired) for reference
PD={}
for a in ["fast","ours","ideal"]:
    PD[a]=[float(np.mean([1-(F[a][s][c:c+3]*F["base"][s][c:c+3]).sum(-1).mean() for s in F[a] if s in F["base"]])) for c in chunks]
# ---- PCA on all frames
allX=np.concatenate([F[a][s] for a in ARMS for s in F[a]]); mu=allX.mean(0); U,Sv,Vt=np.linalg.svd(allX-mu,full_matrices=False); P2=lambda X:(X-mu)@Vt[:2].T
fig,axes=plt.subplots(1,4,figsize=(22,5.5),sharex=True,sharey=True)
for ax,a in zip(axes,ARMS):
    for s in F[a]:
        p=P2(F[a][s]); ax.plot(p[:,0],p[:,1],color=ARMS[a][1],alpha=.35,lw=.8); ax.scatter(p[:,0],p[:,1],c=np.arange(T),cmap="Greys",s=6,alpha=.6)
    # base cloud in the background of every panel
    if a!="base":
        for s in F["base"]: p=P2(F["base"][s]); ax.plot(p[:,0],p[:,1],color="tab:blue",alpha=.12,lw=.6)
    ax.set_title("%s (n=%d seeds)"%(ARMS[a][0],len(F[a])),fontsize=10); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("S05 — DINOv2 trajectories across seeds, one PCA (grey = time light→dark; blue background = Base seeds)",fontsize=12); fig.tight_layout(); fig.savefig(O+"/pca_traj.png",dpi=110); plt.close(fig)
t=np.array(chunks)*3/16*4/4; t=np.array(chunks)/4.0
fig,axes=plt.subplots(1,3,figsize=(20,5))
ax=axes[0]
for a,lab in [("fast","Fast vs Base"),("ours","ours vs Base"),("ideal","ideal vs Base"),("base_split","Base half vs Base half (floor)")]: ax.plot(t,ED[a],marker="o",label=lab,color={"fast":"tab:red","ours":"tab:green","ideal":"tab:purple","base_split":"gray"}[a])
ax.axvspan(4.5,4.75,color="gold",alpha=.3); ax.set_xlabel("time (s)"); ax.set_ylabel("energy distance (DINO 1-cos)"); ax.set_title("distribution distance to Base seeds, per chunk ↓"); ax.legend(fontsize=8)
ax=axes[1]
for a in ARMS: ax.plot(t,SP[a],marker="o",label=ARMS[a][0],color=ARMS[a][1])
ax.axvspan(4.5,4.75,color="gold",alpha=.3); ax.set_xlabel("time (s)"); ax.set_ylabel("mean pairwise distance between seeds"); ax.set_title("spread across seeds (collapse vs diversity)"); ax.legend(fontsize=8)
ax=axes[2]
for a in ["fast","ours","ideal"]: ax.plot(t,PD[a],marker="o",label=ARMS[a][0],color=ARMS[a][1])
ax.axvspan(4.5,4.75,color="gold",alpha=.3); ax.set_xlabel("time (s)"); ax.set_ylabel("DINO 1-cos to same-seed Base"); ax.set_title("paired distance to the matched Base seed ↓"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(O+"/energy_by_chunk.png",dpi=110); plt.close(fig)
late=slice(6,13); summ={a:dict(n=n[a]) for a in ARMS}
for a in ["fast","ours","ideal"]: summ[a].update(energy_all=float(np.mean(ED[a])), energy_late=float(np.mean(ED[a][6:])), paired_late=float(np.mean(PD[a][6:])), spread_late=float(np.mean(SP[a][6:])))
summ["base"].update(energy_floor_late=float(np.mean(ED["base_split"][6:])), spread_late=float(np.mean(SP["base"][6:])))
json.dump(dict(summary=summ,ED=ED,SP=SP,PD=PD,chunks=chunks),open(O+"/summary.json","w"),indent=1)
md="# S05 trajectory distributions across seeds (DINOv2, 4 fps; chunks 6–12 = after the 4.5 s correction point)\n\n| arm | seeds | energy distance to Base seeds, chunks 6–12 ↓ | paired distance to same-seed Base ↓ | spread across seeds (Base = %.3f) |\n|---|---|---|---|---|\n"%summ["base"]["spread_late"]
md+="| Base half vs half (floor) | %d | %.4f | – | %.3f |\n"%(n["base"],summ["base"]["energy_floor_late"],summ["base"]["spread_late"])
for a in ["fast","ours","ideal"]: md+="| %s | %d | %.4f | %.3f | %.3f |\n"%(ARMS[a][0],n[a],summ[a]["energy_late"],summ[a]["paired_late"],summ[a]["spread_late"])
open(O+"/summary.md","w").write(md); print(md)
