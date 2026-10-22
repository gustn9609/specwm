#!/usr/bin/env python3
"""Base-likeness axis, displayed-stream view: for ours/ideal the viewer sees Fast frames until the 4.5 s switch (solid), the KV content (target / Base) is drawn dotted before the switch."""
import os, json, glob, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S="/home/hk28965/specwm"; R=S+"/lingbot-world-v2/specwm_out"; O=R+"/spec_dist/S05"; torch.hub.set_dir(S+"/.torch_hub")
m=torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).eval().cuda(); mean=torch.tensor([0.485,0.456,0.406]).view(3,1,1).cuda(); std=torch.tensor([0.229,0.224,0.225]).view(3,1,1).cuda()
def feats(V):
    X=torch.from_numpy(np.ascontiguousarray(V)).permute(0,3,1,2).float().cuda()/255; X=torch.nn.functional.interpolate(X,size=(518,518),mode="bilinear"); X=(X-mean)/std; o=[]
    with torch.no_grad():
        for i in range(0,len(X),16): o.append(torch.nn.functional.normalize(m(X[i:i+16]).flatten(1),dim=-1).cpu())
    return torch.cat(o).numpy()
ARMS={"base":("Base 70-step (target)","tab:blue"),"fast":("Fast 4-step (draft)","tab:red"),"ours":("Fast + ours (KV-rewrite @4.5 s)","tab:green"),"ideal":("Fast + ideal (Base KV @4.5 s)","tab:purple")}
EXTRA={"every2":("ideal periodic: Base ctx every 1.5 s","tab:cyan"),"every3":("ideal periodic: every 2.25 s","tab:olive"),"every4":("ideal periodic: every 3 s","tab:brown"),"kvrw_second_k10":("ours ×2 (2nd KV-rewrite @7.5 s)","darkgreen")}
seeds=sorted({int(d.split("_s")[-1]) for d in glob.glob(R+"/floor/base/S05_s*")}); F={a:{} for a in list(ARMS)+list(EXTRA)}
for sd in seeds:
    P={"base":R+"/floor/base/S05_s%d"%sd,"fast":R+"/spec_dist/S05/W18_sink6/fast_alone_s%d"%sd,"ours":R+"/spec_dist/S05/W18_sink6/kvrw_laundered_s%d"%sd,"ideal":R+"/spec_dist/S05/W18_sink6/resync_k6_s%d"%sd}
    for e in EXTRA: P[e]=R+"/spec_dist/S05/W18_sink6/%s_s%d"%(e,sd)
    if not os.path.exists(P["fast"]+"/frames_all.npy"): P["fast"]=R+"/floor/fast/S05/W18_sink6/fast_alone_s%d"%sd
    for a,p in P.items():
        if os.path.exists(p+"/frames_all.npy"): F[a][sd]=feats(np.load(p+"/frames_all.npy"))[:39]
T=39; chunks=list(range(0,T,3)); t=np.array(chunks)/4.0
XB=np.concatenate([F["base"][s][18:] for s in F["base"]]); XF=np.concatenate([F["fast"][s][18:] for s in F["fast"]]); mb,mf=XB.mean(0),XF.mean(0); Sw=np.cov(np.concatenate([XB-mb,XF-mf]).T)+1e-3*np.eye(XB.shape[1]); w=np.linalg.solve(Sw,mb-mf); w/=np.linalg.norm(w)
def proj(a): return {s:np.array([F[a][s][c:c+3].mean(0)@w for c in chunks]) for s in F[a]}
PR={a:proj(a) for a in F if F[a]}
def med(a,idx=None): M=np.stack(list(PR[a].values())); return np.median(M,0), np.percentile(M,25,0), np.percentile(M,75,0)
fig,axes=plt.subplots(1,2,figsize=(20,6),sharey=True)
# left: displayed stream (single correction)
ax=axes[0]
for a in ["base","fast"]:
    md,q1,q3=med(a); ax.plot(t,md,color=ARMS[a][1],lw=2.2,marker="o",label=ARMS[a][0]); ax.fill_between(t,q1,q3,color=ARMS[a][1],alpha=.15)
fm=med("fast")[0]
for a in ["ours","ideal"]:
    md,q1,q3=med(a); sw=6   # chunk index of the switch
    disp=np.concatenate([fm[:sw],md[sw:]]); ax.plot(t[sw:],md[sw:],color=ARMS[a][1],lw=2.2,marker="o",label=ARMS[a][0]+" — displayed after switch"); ax.fill_between(t[sw:],q1[sw:],q3[sw:],color=ARMS[a][1],alpha=.15)
    ax.plot(t[:sw+1],md[:sw+1],color=ARMS[a][1],lw=1.2,ls=":",marker=".",label=("KV content before the switch (target / Base), not displayed" if a=="ours" else None))
    ax.plot([t[sw-1],t[sw]],[fm[sw-1],md[sw]],color=ARMS[a][1],lw=1,ls="--",alpha=.7)   # the visible cut
ax.axvspan(4.5,4.75,color="gold",alpha=.3); ax.text(4.6,ax.get_ylim()[1]*0.98,"switch (visible cut)",ha="center",va="top",fontsize=8,color="darkgoldenrod")
ax.set_xlabel("time (s)"); ax.set_ylabel("Base-likeness (projection on Base-vs-Fast LDA axis)  →"); ax.set_title("what the viewer sees: Fast until 4.5 s, then generation on the rewritten KV (median, IQR)",fontsize=10); ax.legend(fontsize=8,loc="lower left")
# right: periodic arms (if present)
ax=axes[1]
for a in ["base","fast","ours","ideal"]:
    md,q1,q3=med(a); ax.plot(t,md,color=ARMS[a][1],lw=1.6,marker="o",ms=3,alpha=.8,label=ARMS[a][0])
for e in EXTRA:
    if e in PR and PR[e]:
        md,q1,q3=med(e); ax.plot(t,md,color=EXTRA[e][1],lw=2.2,marker="s",ms=4,label=EXTRA[e][0]+" (n=%d)"%len(PR[e])); ax.fill_between(t,q1,q3,color=EXTRA[e][1],alpha=.12)
ax.axvspan(4.5,4.75,color="gold",alpha=.3); ax.axvspan(7.5,7.75,color="gold",alpha=.2)
ax.set_xlabel("time (s)"); ax.set_title("periodic correction (saved streams; dotted-style caveat applies before each re-injection)",fontsize=10); ax.legend(fontsize=7,loc="lower left")
fig.suptitle("S05, %d seeds — Base-likeness over time"%len(F["base"]),fontsize=12); fig.tight_layout(); fig.savefig(O+"/baselike_displayed.png",dpi=110); plt.close(fig)
summ={a:{"median_by_chunk":[float(x) for x in med(a)[0]],"n":len(PR[a])} for a in PR if PR[a]}; json.dump(summ,open(O+"/baselike_summary.json","w"),indent=1); print({a:summ[a]["n"] for a in summ})
