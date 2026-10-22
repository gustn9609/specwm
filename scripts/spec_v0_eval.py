#!/usr/bin/env python3
"""spec v0 evaluation on the two failure tasks: DINOv2 distance of each arm/seed to the task's Base 70-step seed batch, per chunk + strips.
usage: python scripts/spec_v0_eval.py [--judge-dir DIR]  -> specwm_out/spec_v0/eval/{summary.md,summary.json,dist_<task>.png,strip_<task>.jpg}"""
import os, sys, glob, json, argparse, numpy as np, torch, imageio, cv2
S="/home/hk28965/specwm"; R=S+"/lingbot-world-v2/specwm_out/spec_v0"; O=R+"/eval"; os.makedirs(O,exist_ok=True); torch.hub.set_dir(S+"/.torch_hub")
TASKS=["sand_filling_up_jar","sandcastle_wave_impact"]; ARMS0=["never","verify_999","every_3","every_2","always"]; NAMES=dict(never="Fast 14B (draft only)",verify_999="verify:999 (=Fast, logs d)",every_3="every:3 (4 corr.)",every_2="every:2 (6 corr.)",always="always (12 corr.)",base70="Base 70-step")
class _N(dict):
    def __getitem__(self, k): return dict.get(self, k, k)
NAMES=_N(NAMES)
m=torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).eval().cuda(); mean=torch.tensor([0.485,0.456,0.406]).view(3,1,1).cuda(); std=torch.tensor([0.229,0.224,0.225]).view(3,1,1).cuda()
def load4(p):   # frames at 4 fps (first 161 f -> 41 frames)
    V=[np.asarray(f) for f in imageio.get_reader(p)][:161]; return np.stack(V[::4])
@torch.no_grad()
def feats(V):
    out=[]
    for i in range(0,len(V),16):
        x=torch.from_numpy(V[i:i+16]).cuda().permute(0,3,1,2).float()/255; x=torch.nn.functional.interpolate(x,size=(224,392),mode="bilinear",antialias=True); x=(x-mean)/std
        f=m(x); out.append(torch.nn.functional.normalize(f,dim=-1).cpu().numpy())
    return np.concatenate(out)
res={}; F={}; DIRS={}
for t in TASKS:
    F[t]={}
    for d in sorted(glob.glob(R+"/"+t+"/*_s*")):
        p=d+"/video.mp4"
        if not os.path.exists(p) or not os.path.basename(d).rsplit("_s",1)[-1].isdigit(): continue
        F[t][os.path.basename(d)]=feats(load4(p)); DIRS[os.path.basename(d)]=d
    base=[F[t][k] for k in F[t] if k.startswith("base70")]
    res[t]={}
    for k,f in F[t].items():
        arm,sd=k.rsplit("_s",1); n=min(len(f),min(len(b) for b in base))
        # per-frame distance to Base batch = mean over Base seeds of (1-cos) at the same time index; Base seeds compared leave-one-out
        others=[b for b in base if b is not f]
        D=np.mean([1-np.sum(f[:n]*b[:n],1) for b in others],0)
        chunks=[D[max(0,3*c-1):3*c+2].mean() for c in range(13)]   # ~chunk c covers frames 3c-1..3c+1 (0.75 s)
        res[t][k]=dict(arm=arm,seed=int(sd),all=float(D.mean()),late=float(D[12:].mean()),chunks=[float(x) for x in chunks])
        j=DIRS[k]+"/spec_v0.json"
        if os.path.exists(j): jj=json.load(open(j)); res[t][k].update(gen_s=jj["gen_s"],base_s=jj["base_s"],plan=jj["plan"],t_denoise=jj.get("t_denoise"),t_swap=jj.get("t_swap"),t_cond=jj.get("t_cond"),nodes=[l.get("nodes") for l in jj["log"]])
json.dump(res,open(O+"/summary.json","w"),indent=1)
ARMS=ARMS0+sorted({res[t][k]["arm"] for t in TASKS for k in res[t]}-set(ARMS0)-{"base70"})
# ---- tables
md="# spec v0 on the two failure tasks — DINOv2 (ViT-B/14 CLS, 1−cos, 4 fps) distance to the task's Base 70-step seed batch\n\nBase row = leave-one-out among Base seeds (floor). late = 3–10 s (frames 12–40). Values: mean over seeds (n).\n\n"
for t in TASKS:
    md+="## %s\n\n| arm | n | DINO to Base, all ↓ | DINO to Base, 3–10 s ↓ | gen time (s) ↓ | Base time (s) ↓ | Base corrections |\n|---|---|---|---|---|---|---|\n"%t
    rows=[]
    for arm in ["base70"]+ARMS:
        ks=[k for k in res[t] if res[t][k]["arm"]==arm]
        if not ks: continue
        rows.append((arm,len(ks),np.mean([res[t][k]["all"] for k in ks]),np.mean([res[t][k]["late"] for k in ks]),np.mean([res[t][k].get("gen_s",np.nan) for k in ks]),np.mean([res[t][k]["plan"].count("C") if "plan" in res[t][k] else 0 for k in ks]),np.mean([res[t][k].get("base_s",np.nan) for k in ks])))
    def fmt(v,col,best,second,f="%.3f"):
        s=f%v; return "**%s**"%s if v==best else ("<u>%s</u>"%s if v==second else s)
    for ci,f in [(2,"%.3f"),(3,"%.3f"),(4,"%.0f")]:
        vals=sorted(set(r[ci] for r in rows[1:] if not np.isnan(r[ci])))   # best among non-Base arms
        for r in rows:
            pass
    for r in rows:
        cells=[]
        for ci,f in [(2,"%.3f"),(3,"%.3f"),(4,"%.0f")]:
            vals=sorted(set(x[ci] for x in rows[1:] if not np.isnan(x[ci]))); best=vals[0] if vals else None; second=vals[1] if len(vals)>1 else None
            cells.append(f%r[ci] if r[0]=="base70" or np.isnan(r[ci]) else fmt(r[ci],ci,best,second,f))
        md+="| %s | %d | %s | %s | %s | %s | %.0f |\n"%(NAMES[r[0]],r[1],cells[0],cells[1],cells[2],"–" if np.isnan(r[6]) else "%.0f"%r[6],r[5])
    md+="\n"
# ---- per-chunk distance plot + strips
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
COL=_N(dict(base70="k",never="tab:red",verify_999="tab:pink",every_3="tab:orange",every_2="tab:green",always="tab:blue"))
import itertools; _cyc=itertools.cycle(plt.cm.tab20.colors)
for arm in ARMS:
    if arm not in COL: COL[arm]=next(_cyc)
for t in TASKS:
    fig,ax=plt.subplots(figsize=(8,3.6))
    for arm in ["base70"]+ARMS:
        ks=[k for k in res[t] if res[t][k]["arm"]==arm]
        if not ks: continue
        C=np.array([res[t][k]["chunks"] for k in ks]); x=np.arange(13)*0.75
        ax.plot(x,C.mean(0),color=COL[arm],marker="o",ms=3,label="%s (n=%d)"%(NAMES[arm],len(ks)))
        if len(ks)>1: ax.fill_between(x,C.min(0),C.max(0),color=COL[arm],alpha=.12)
    ax.set_xlabel("time (s), chunk = 0.75 s"); ax.set_ylabel("DINO 1−cos to Base seeds ↓"); ax.set_title(t); ax.legend(fontsize=7); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(O+"/dist_%s.png"%t,dpi=120); plt.close(fig)
    rows=[]
    for k in sorted(res[t],key=lambda k:(["base70"]+ARMS).index(res[t][k]["arm"])*10+res[t][k]["seed"]%7):
        V=load4(R+"/"+t+"/"+k+"/video.mp4"); ks=np.linspace(0,len(V)-1,13).round().astype(int); tiles=[cv2.resize(V[i],(160,90)) for i in ks]
        for i,im in zip(ks,tiles): cv2.putText(im,"%.1fs"%(i/4),(3,12),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),2)
        strip=np.concatenate(tiles,1); lbl=np.full((20,strip.shape[1],3),245,np.uint8); cv2.putText(lbl,"%s  seed %d  DINO late %.3f"%(NAMES[res[t][k]["arm"]],res[t][k]["seed"],res[t][k]["late"]),(4,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(30,30,30),1); rows+=[lbl,strip]
    imageio.imwrite(O+"/strip_%s.jpg"%t,np.concatenate(rows,0),quality=80)
open(O+"/summary.md","w").write(md); print(md)
