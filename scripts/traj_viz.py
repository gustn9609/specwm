#!/usr/bin/env python3
"""Base vs Fast trajectory visualization.
per clip pair: (1) DINO feature trajectory (PCA-2D, fit on both) with time coloring, (2) drift curves: Fast_t vs Base_t distance, Fast frame-to-frame jump,
Base frame-to-frame jump, (3) frame strip at 1 s intervals (Base row / Fast row) with the largest-jump frames boxed. Output: traj_viz/<name>.png + summary json."""
import os, sys, json, glob, numpy as np, torch, imageio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S="/home/hk28965/specwm"; O=S+"/lingbot-world-v2/specwm_out"; OUT=O+"/traj_viz"; os.makedirs(OUT, exist_ok=True)
torch.hub.set_dir(S+"/.torch_hub"); dev="cuda"
m=torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).eval().to(dev)
mean=torch.tensor([0.485,0.456,0.406]).view(3,1,1).to(dev); std=torch.tensor([0.229,0.224,0.225]).view(3,1,1).to(dev)
def load_frames(p, fps_target=4):
    if p.endswith(".npy"): V=np.load(p); V=V[::4] if len(V)>100 else V; return V
    rd=imageio.get_reader(p); fps=rd.get_meta_data().get("fps",16); step=max(int(round(fps/fps_target)),1)
    return np.stack([f for i,f in enumerate(rd) if i%step==0])
def feats(V):
    X=torch.from_numpy(np.ascontiguousarray(V)).permute(0,3,1,2).float().to(dev)/255
    X=torch.nn.functional.interpolate(X,size=(518,518),mode="bilinear"); X=(X-mean)/std; o=[]
    with torch.no_grad():
        for i in range(0,len(X),8): o.append(torch.nn.functional.normalize(m(X[i:i+8]).flatten(1),dim=-1).cpu())
    return torch.cat(o).numpy()
def viz(name, base_p, fast_p, title, fps=4, note=""):
    VB, VF = load_frames(base_p), load_frames(fast_p); n=min(len(VB),len(VF)); VB,VF=VB[:n],VF[:n]; FB,FF=feats(VB),feats(VF)
    dBF=1-(FB*FF).sum(-1); jF=np.r_[0,1-(FF[1:]*FF[:-1]).sum(-1)]; jB=np.r_[0,1-(FB[1:]*FB[:-1]).sum(-1)]; d0F=1-(FF*FF[0:1]).sum(-1); d0B=1-(FB*FB[0:1]).sum(-1)
    Xall=np.concatenate([FB,FF]); Xc=Xall-Xall.mean(0); U,Sv,Vt=np.linalg.svd(Xc,full_matrices=False); P=Xc@Vt[:2].T; PB,PF=P[:n],P[n:]
    t=np.arange(n)/fps; top=np.argsort(jF)[-3:][::-1]                       # Fast 의 프레임 간 점프 상위 3
    fig=plt.figure(figsize=(20,9.5)); gs=fig.add_gridspec(2,3,height_ratios=[1,0.55])
    away=None
    if "stevo" in name: away=(3*4/16.0, 33*4/16.0)         # StEvo pan: latents 3..33 away, 33..63 back (4 px f per latent @16fps)
    ax=fig.add_subplot(gs[0,0])
    for Pq,lab,cm,mk in [(PB,"Base (70-step)","Blues","o"),(PF,"Fast (4-step AR)","Reds","s")]:
        ax.plot(Pq[:,0],Pq[:,1],color=plt.get_cmap(cm)(0.6),lw=1,alpha=.6); ax.scatter(Pq[:,0],Pq[:,1],c=t,cmap=cm,s=30,marker=mk,label=lab,edgecolor="k",linewidths=.3)
        ax.annotate("start",Pq[0],fontsize=8); ax.annotate("end",Pq[-1],fontsize=8)
    for k in range(0,n,max(n//8,1)): ax.plot([PB[k,0],PF[k,0]],[PB[k,1],PF[k,1]],color="gray",lw=.5,ls=":")
    ax.set_title("DINOv2 trajectory (PCA fit on both; light→dark = time; dotted = same t)",fontsize=10); ax.legend(fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    ax=fig.add_subplot(gs[0,1]); ax.plot(t,dBF,"k-",lw=2,label="Fast_t vs Base_t (divergence)"); ax.plot(t,d0B,color="tab:blue",lw=1,label="Base_t vs Base_0"); ax.plot(t,d0F,color="tab:red",lw=1,label="Fast_t vs Fast_0")
    ax.axhline(0.17,color="gray",ls="--",lw=.8); ax.text(t[-1],0.17,"seed floor",fontsize=7,ha="right",va="bottom")
    if away: ax.axvspan(*away,color="gold",alpha=.15); ax.text(sum(away)/2,ax.get_ylim()[1]*0.97,"camera looking away",ha="center",va="top",fontsize=8,color="darkgoldenrod")
    ax.set_xlabel("time (s)"); ax.set_ylabel("DINO 1-cos"); ax.legend(fontsize=7); ax.set_title("drift from Base / from first frame",fontsize=10)
    ax=fig.add_subplot(gs[0,2]); ax.plot(t,jB,color="tab:blue",lw=1.2,label="Base frame-to-frame"); ax.plot(t,jF,color="tab:red",lw=1.2,label="Fast frame-to-frame")
    for k in top: ax.axvline(t[k],color="tab:red",ls=":",lw=.8); ax.annotate("%.2gs"%t[k],(t[k],jF[k]),fontsize=7,color="tab:red")
    if away: ax.axvspan(*away,color="gold",alpha=.15)
    ax.set_xlabel("time (s)"); ax.set_ylabel("DINO 1-cos (Δt=0.25 s)"); ax.legend(fontsize=7); ax.set_title("temporal jumps (spikes = abrupt object/scene change)",fontsize=10)
    ax=fig.add_subplot(gs[1,:]); ax.axis("off"); ks=list(range(0,n,fps)); h,w=VB.shape[1:3]; tw=int(1800/len(ks)); th=int(h*tw/w)
    import cv2; rows=[]
    for V,lab,col in [(VB,"Base",(40,90,200)),(VF,"Fast",(200,50,50))]:
        tiles=[]
        for k in ks:
            im=np.ascontiguousarray(cv2.resize(V[k],(tw,th)))
            if V is VF and any(abs(k-kk)<=1 for kk in top): im=cv2.rectangle(im,(2,2),(tw-3,th-3),(255,0,0),4)
            if away and away[0]<=k/fps<=away[1]: im=cv2.rectangle(im,(0,0),(tw-1,th-1),(230,190,0),2)
            cv2.putText(im,"%ds"%(k//fps),(5,16),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2); tiles.append(im)
        strip=np.concatenate(tiles,1); lbl=np.full((24,strip.shape[1],3),245,np.uint8); cv2.putText(lbl,lab,(6,18),cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2); rows+=[lbl,strip]
    ax.imshow(np.concatenate(rows,0)); ax.set_title("frames at 1 s intervals — red box: Fast frame at a top-3 temporal jump"+("; gold border: camera looking away" if away else "")+("\n"+note if note else ""),fontsize=9)
    fig.suptitle(title,fontsize=13); fig.tight_layout(); fig.savefig(OUT+"/%s.png"%name,dpi=105); plt.close(fig)
    rep=dict(name=name,n=int(n),mean_divergence=float(dBF.mean()),late_divergence=float(dBF[n//2:].mean()),fast_jump_max=float(jF.max()),fast_jump_at_s=float(t[jF.argmax()]),base_jump_max=float(jB.max()),fast_top_jumps_s=[float(t[k]) for k in top]); print(rep,flush=True); return rep
reps=[]
for sc,nm in [("05","dragon (S05)"),("01","stonehenge (S01)"),("00","great wall (S00)"),("02","urban (S02)")]:
    sd="1234"; reps.append(viz("floor_S%s_s%s"%(sc,sd), O+"/floor/base/S%s_s%s/frames_all.npy"%(sc,sd), O+"/floor/fast/S%s/W18_sink6/fast_alone_s%s/frames_all.npy"%(sc,sd), "%s — seed %s, same camera path & prompt, 10 s"%(nm,sd)))
for tid,note in [("falling_glass","StEvo: wine glass on table edge; camera pans away and back"),("pouring_water_into_cup","StEvo: pitcher pouring; camera pans away and back"),("rocket_launch","StEvo: rocket launch; camera pans away and back"),("inflating_balloon","StEvo: balloon inflating; camera pans away and back")]:
    reps.append(viz("stevo_"+tid, O+"/stevo/lingbot_base/%s/video.mp4"%tid, O+"/stevo/lingbot_fast/%s.mp4"%tid, "StEvo %s — Base (physics OK per judge) vs Fast (flagged)"%tid, note=note))
json.dump(reps,open(OUT+"/summary.json","w"),indent=1); print("[TRAJ_DONE]")
