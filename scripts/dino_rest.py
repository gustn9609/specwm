import os
import torch, numpy as np, json, os, itertools
torch.set_num_threads(4); torch.hub.set_dir(os.environ["SCRATCH"]+"/.torch_hub")
m = torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).eval()
mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
R=os.environ["SCRATCH"]+"/lingbot-world-v2/specwm_out"
@torch.no_grad()
def feats(V,idx,bs=2):
    o=[]
    for i in range(0,len(idx),bs):
        b=torch.from_numpy(V[idx[i:i+bs]]).permute(0,3,1,2).float()/255.
        b=torch.nn.functional.interpolate(b,size=(518,518),mode="bicubic",align_corners=False)
        o.append(torch.nn.functional.normalize(m(((b-mean)/std)).flatten(1).float(),dim=-1))
    return torch.cat(o)
def dist(A,B,idx): n=min(len(A),len(B)); ii=[i for i in idx if i<n]; return float(np.mean([1-float(A[i]@B[i]) for i in ii]))
out={}
# ---- 1) 공식 장면 ablation (붕괴 안 한 arm 만) ----
print("== ablation (official scenes) ==",flush=True)
for sc,ref,arms in [("S05",R+"/revisit/base_official_S05/frames_all.npy",{"base16":R+"/revisit/base_official_S05_s16/frames_all.npy","base4":R+"/revisit/base_official_S05_s4/frames_all.npy","fast":R+"/ablate2/S05_fast/W18_sink6/fast_alone_s1234/frames_all.npy"}),
                    ("S01",R+"/revisit/base_official_S01/frames_all.npy",{"base16":R+"/ablate2/S01_base16/frames_all.npy","base4":R+"/ablate2/S01_base4/frames_all.npy","fast":R+"/ablate2/S01_fast/W18_sink6/fast_alone_s1234/frames_all.npy"})]:
    Vr=np.load(ref); Fr=feats(Vr,list(range(len(Vr))))
    for a,p in arms.items():
        V=np.load(p); F=feats(V,list(range(len(V)))); n=min(len(F),len(Fr))
        whole=dist(F,Fr,range(n)); last=dist(F,Fr,range(n-5,n)); out["%s_%s"%(sc,a)]=dict(whole=round(whole,4),last5=round(last,4),growth=round(last-whole,4))
        print("  %s %-7s whole %.4f last5 %.4f growth %+.4f"%(sc,a,whole,last,last-whole),flush=True)
# ---- 2) mask vs RePaint 다운스트림 ----
print("== provider: mask vs RePaint ==",flush=True)
P=R+"/ablate2/provider/W18_sink6"; Vr=np.load(R+"/revisit/base_official_S05_s16/frames_all.npy"); Fr=feats(Vr,list(range(len(Vr))))
after=list(range(18,len(Vr)))
pv={}
for a in ["fast_alone","kvrw_repaint","kvrw_mask"]:
    p="%s/%s_s1234/frames_all.npy"%(P,a)
    if not os.path.exists(p): print("  MISS",a); continue
    V=np.load(p); F=feats(V,list(range(len(V)))); pv[a]=dist(F,Fr,after); print("  %-13s after chunk6: %.4f"%(a,pv[a]),flush=True)
if "fast_alone" in pv:
    for a in ["kvrw_repaint","kvrw_mask"]:
        if a in pv: print("  %-13s gain %+.4f"%(a,pv["fast_alone"]-pv[a]))
out["provider"]=pv
# ---- 3) 시드 3개 지연/주기 ----
print("== latency / period, 3 seeds ==",flush=True)
U=np.load(R+"/resync55/W36_sink18/resync_k6_s1234/frames_all.npy"); Fu=feats(U,list(range(180,219,2)))
late=list(range(180,219,2))
def d55(p):
    if not os.path.exists(p): return None
    V=np.load(p); F=feats(V,late); return float(np.mean([1-float(F[j]@Fu[j]) for j in range(len(late)) if late[j]<len(V)]))
arms={"control":{1234:R+"/e2e55/control/frames_all.npy",4321:R+"/lat55_s4321/control/frames_all.npy",9999:R+"/lat55_s9999/control/frames_all.npy"},
      "d0":{1234:R+"/lat55/d0/frames_all.npy",4321:R+"/lat55_s4321/d0/frames_all.npy",9999:R+"/lat55_s9999/d0/frames_all.npy"},
      "d3":{1234:R+"/lat55/d3/frames_all.npy",4321:R+"/lat55_s4321/d3/frames_all.npy",9999:R+"/lat55_s9999/d3/frames_all.npy"},
      "d7":{1234:R+"/lat55/d7/frames_all.npy",4321:R+"/lat55_s4321/d7/frames_all.npy",9999:R+"/lat55_s9999/d7/frames_all.npy"},
      "every30":{1234:R+"/per55/every30/frames_all.npy",4321:R+"/per55_s4321/every30/frames_all.npy",9999:R+"/per55_s9999/every30/frames_all.npy"}}
S3={}
for a,ps in arms.items():
    vals={sd:d55(p) for sd,p in ps.items()}; v=[x for x in vals.values() if x is not None]
    S3[a]=dict(per_seed={str(k):(round(x,4) if x else None) for k,x in vals.items()},mean=round(float(np.mean(v)),4),std=round(float(np.std(v,ddof=1)) if len(v)>1 else 0,4),n=len(v))
    print("  %-8s %s  mean %.4f ± %.4f (n=%d)"%(a,{k:(round(x,3) if x else None) for k,x in vals.items()},S3[a]["mean"],S3[a]["std"],len(v)),flush=True)
out["seeds3"]=S3
json.dump(out,open(os.environ["SCRATCH"]+"/rest_dino.json","w"),indent=1); print("DONE",flush=True)
