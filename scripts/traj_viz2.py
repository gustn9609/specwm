#!/usr/bin/env python3
"""Base vs Fast trajectory visualization.
per clip pair: (1) DINO feature trajectory (PCA-2D, fit on both) with time coloring, (2) drift curves: Fast_t vs Base_t distance, Fast frame-to-frame jump,
Base frame-to-frame jump, (3) frame strip at 1 s intervals (Base row / Fast row) with the largest-jump frames boxed. Output: traj_viz/<name>.png + summary json."""
import os, sys, json, glob, numpy as np, torch, imageio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S="/home/hk28965/specwm"; O=S+"/lingbot-world-v2/specwm_out"; OUT=O+"/traj_viz2"; os.makedirs(OUT, exist_ok=True)
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
def viz(name, base_p, fast_p, title, fps=4, note="", lab_ref="Base (70-step)", lab_cmp="Fast (4-step AR)", away_on=True):
    VB, VF = load_frames(base_p), load_frames(fast_p); n=min(len(VB),len(VF)); VB,VF=VB[:n],VF[:n]; FB,FF=feats(VB),feats(VF)
    dBF=1-(FB*FF).sum(-1); jF=np.r_[0,1-(FF[1:]*FF[:-1]).sum(-1)]; jB=np.r_[0,1-(FB[1:]*FB[:-1]).sum(-1)]; d0F=1-(FF*FF[0:1]).sum(-1); d0B=1-(FB*FB[0:1]).sum(-1)
    Xall=np.concatenate([FB,FF]); Xc=Xall-Xall.mean(0); U,Sv,Vt=np.linalg.svd(Xc,full_matrices=False); P=Xc@Vt[:2].T; PB,PF=P[:n],P[n:]
    t=np.arange(n)/fps; top=np.argsort(jF)[-3:][::-1]                       # Fast 의 프레임 간 점프 상위 3
    fig=plt.figure(figsize=(20,9.5)); gs=fig.add_gridspec(2,3,height_ratios=[1,0.55])
    away=None
    if "stevo" in name and away_on: away=(3*4/16.0, 33*4/16.0)         # StEvo pan: latents 3..33 away, 33..63 back (4 px f per latent @16fps)
    ax=fig.add_subplot(gs[0,0])
    for Pq,lab,cm,mk in [(PB,lab_ref,"Blues","o"),(PF,lab_cmp,"Reds","s")]:
        ax.plot(Pq[:,0],Pq[:,1],color=plt.get_cmap(cm)(0.6),lw=1,alpha=.6); ax.scatter(Pq[:,0],Pq[:,1],c=t,cmap=cm,s=30,marker=mk,label=lab,edgecolor="k",linewidths=.3)
        ax.annotate("start",Pq[0],fontsize=8); ax.annotate("end",Pq[-1],fontsize=8)
    for k in range(0,n,max(n//8,1)): ax.plot([PB[k,0],PF[k,0]],[PB[k,1],PF[k,1]],color="gray",lw=.5,ls=":")
    ax.set_title("DINOv2 trajectory (PCA fit on both; light→dark = time; dotted = same t)",fontsize=10); ax.legend(fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    ax=fig.add_subplot(gs[0,1]); ax.plot(t,dBF,"k-",lw=2,label="%s_t vs %s_t (divergence)"%(lab_cmp.split(" ")[0],lab_ref.split(" ")[0])); ax.plot(t,d0B,color="tab:blue",lw=1,label="%s_t vs its first frame"%lab_ref.split(" ")[0]); ax.plot(t,d0F,color="tab:red",lw=1,label="%s_t vs its first frame"%lab_cmp.split(" ")[0])
    ax.axhline(0.17,color="gray",ls="--",lw=.8); ax.text(t[-1],0.17,"seed floor",fontsize=7,ha="right",va="bottom")
    if away: ax.axvspan(*away,color="gold",alpha=.15); ax.text(sum(away)/2,ax.get_ylim()[1]*0.97,"camera looking away",ha="center",va="top",fontsize=8,color="darkgoldenrod")
    ax.set_xlabel("time (s)"); ax.set_ylabel("DINO 1-cos"); ax.legend(fontsize=7); ax.set_title("drift from Base / from first frame",fontsize=10)
    ax=fig.add_subplot(gs[0,2]); ax.plot(t,jB,color="tab:blue",lw=1.2,label="%s frame-to-frame"%lab_ref.split(" ")[0]); ax.plot(t,jF,color="tab:red",lw=1.2,label="%s frame-to-frame"%lab_cmp.split(" ")[0])
    for k in top: ax.axvline(t[k],color="tab:red",ls=":",lw=.8); ax.annotate("%.2gs"%t[k],(t[k],jF[k]),fontsize=7,color="tab:red")
    if away: ax.axvspan(*away,color="gold",alpha=.15)
    ax.set_xlabel("time (s)"); ax.set_ylabel("DINO 1-cos (Δt=0.25 s)"); ax.legend(fontsize=7); ax.set_title("temporal jumps (spikes = abrupt object/scene change)",fontsize=10)
    ax=fig.add_subplot(gs[1,:]); ax.axis("off"); ks=list(range(0,n,fps)); h,w=VB.shape[1:3]; tw=int(1800/len(ks)); th=int(h*tw/w)
    import cv2; rows=[]
    for V,lab,col in [(VB,lab_ref,(40,90,200)),(VF,lab_cmp,(200,50,50))]:
        tiles=[]
        for k in ks:
            im=np.ascontiguousarray(cv2.resize(V[k],(tw,th)))
            if V is VF and any(abs(k-kk)<=1 for kk in top): im=cv2.rectangle(im,(2,2),(tw-3,th-3),(255,0,0),4)
            if away and away[0]<=k/fps<=away[1]: im=cv2.rectangle(im,(0,0),(tw-1,th-1),(230,190,0),2)
            cv2.putText(im,"%ds"%(k//fps),(5,16),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2); tiles.append(im)
        strip=np.concatenate(tiles,1); lbl=np.full((24,strip.shape[1],3),245,np.uint8); cv2.putText(lbl,lab,(6,18),cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2); rows+=[lbl,strip]
    ax.imshow(np.concatenate(rows,0)); ax.set_title("frames at 1 s intervals — red box: %s frame at a top-3 temporal jump"%lab_cmp.split(" ")[0]+("; gold border: camera looking away" if away else "")+("\n"+note if note else ""),fontsize=9)
    fig.suptitle(title,fontsize=13); fig.tight_layout(); fig.savefig(OUT+"/%s.jpg"%name,dpi=80,pil_kwargs={"quality":78}); plt.close(fig)
    rep=dict(name=name,n=int(n),mean_divergence=float(dBF.mean()),late_divergence=float(dBF[n//2:].mean()),fast_jump_max=float(jF.max()),fast_jump_at_s=float(t[jF.argmax()]),base_jump_max=float(jB.max()),fast_top_jumps_s=[float(t[k]) for k in top]); print(rep,flush=True); return rep
reps=[]
ST=O+"/stevo"; TASKS=[("falling_glass","wine glass on table edge"),("pouring_water_into_cup","pitcher pouring"),("rocket_launch","rocket launch"),("inflating_balloon","balloon inflating")]
ARMS=[("fast480","Fast 480p 4-step",lambda t: ST+"/lingbot_fast/%s.mp4"%t,True),("fast360_2step","Fast 360p 2-step",lambda t: ST+"/lingbot_fast_360_2step/%s.mp4"%t,True),("fast360_2step_esrgan","Fast 360p 2-step + ESRGAN",lambda t: ST+"/lingbot_fast_360_2step_esrgan/%s.mp4"%t,True),
      ("v2_1p3b","v2 1.3B causal-fast",lambda t: ST+"/lingbot_v2_1p3b/%s.mp4"%t,True),("v2_pretrain","v2 causal-pretrain 14B",lambda t: ST+"/lingbot_v2_pretrain_360_173/%s.mp4"%t,True),("h3world","H3-World (keyboard pan)",lambda t: ST+"/h3world_lingbotcam/%s.mp4"%t,False),
      ("minimax_h3","MiniMax-H3 (curtain occlusion, no camera)",lambda t: ST+"/h3_video/%s_curtain.mp4"%t,False),("longlive","LongLive-2.0-5B (curtain occlusion, no camera)",lambda t: ST+"/longlive2_5b_video/%s_curtain.mp4"%t,False),
      ("reground_ours","Fast + ours (KV-rewrite, 161 f)",lambda t: ST+"/lingbot_reground_kvrw_laundered/%s.mp4"%t,True),("reground_ideal","Fast + ideal regrounding (161 f)",lambda t: ST+"/lingbot_reground_resync_k6/%s.mp4"%t,True)]
for tid,note in TASKS:
    for key,lab,fn,aw in ARMS:
        p=fn(tid)
        if not os.path.exists(p): print("[skip]",key,tid); continue
        try: reps.append(dict(viz("stevo_%s__%s"%(tid,key), ST+"/lingbot_base/%s/video.mp4"%tid, p, "StEvo %s — Base 70-step vs %s"%(tid,lab), note="StEvo: %s; camera pans away and back (gold)"%note if aw else "StEvo: %s; video-model protocol (occluder), no camera control"%note, lab_ref="Base (70-step)", lab_cmp=lab, away_on=aw), task=tid, arm=key, arm_label=lab))
        except Exception as e: print("[fail]",key,tid,e)
FL=[("05","dragon (S05)"),("01","stonehenge (S01)"),("00","great wall (S00)"),("02","urban (S02)")]
FARMS=[("fast480","Fast 480p 4-step",lambda sc: O+"/floor/fast/S%s/W18_sink6/fast_alone_s1234/frames_all.npy"%sc),("fast360_2step","Fast 360p 2-step draft",lambda sc: O+"/draft360_reground/S%s/W18_sink6/fast_alone_s1234/frames_all.npy"%sc),
       ("reground_ours","Fast + ours (KV-rewrite from laundered target)",lambda sc: O+"/e1repro/S%s/W18_sink6/kvrw_laundered_s1234/frames_all.npy"%sc),("reground_ideal","Fast + ideal regrounding (Base ctx 0-5)",lambda sc: O+"/e1repro/S%s/W18_sink6/resync_k6_s1234/frames_all.npy"%sc),
       ("anchor_restart","Fast restarted from corrected anchor frame (C2)",lambda sc: O+"/c2_anchor/S%s/composite_anchor_target480_s1234.mp4"%sc),("draft360_ours","Fast 360p 2-step + ours (360p target)",lambda sc: O+"/draft360_reground/S%s/W18_sink6/kvrw_laundered_s1234/frames_all.npy"%sc)]
for sc,nm in FL:
    for key,lab,fn in FARMS:
        p=fn(sc)
        if not os.path.exists(p): print("[skip]",key,sc); continue
        try: reps.append(dict(viz("floor_S%s__%s"%(sc,key), O+"/floor/base/S%s_s1234/frames_all.npy"%sc, p, "%s — seed 1234, same camera path & prompt — Base 70-step vs %s"%(nm,lab), lab_ref="Base (70-step)", lab_cmp=lab), task="S"+sc, arm=key, arm_label=lab))
        except Exception as e: print("[fail]",key,sc,e)
json.dump(reps,open(OUT+"/summary.json","w"),indent=1); print("[TRAJ_DONE]")
