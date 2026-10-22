import os
import torch, numpy as np, json, os
torch.set_num_threads(4); torch.hub.set_dir(os.environ["SCRATCH"]+"/.torch_hub")
m = torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).eval()
mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
H=os.environ["SCRATCH"]+"/lingbot-world-v2/specwm_out/hands"
A=["base_official_p1","base_official_p1_s4","base_causal_N3","fast_p1","base_p1"]
@torch.no_grad()
def feats(V,bs=2):
    o=[]
    for i in range(0,len(V),bs):
        b=torch.from_numpy(V[i:i+bs]).permute(0,3,1,2).float()/255.
        b=torch.nn.functional.interpolate(b,size=(518,518),mode="bicubic",align_corners=False)
        o.append(torch.nn.functional.normalize(m(((b-mean)/std)).flatten(1).float(),dim=-1))
    return torch.cat(o)
F={}
for a in A:
    V=np.load(os.path.join(H,a,"frames_all.npy")); F[a]=feats(V)
    print("[%s] npy shape %s -> %d feature vectors of dim %d"%(a,V.shape,len(F[a]),F[a].shape[1]),flush=True)
ref=F["base_official_p1"]; out={}
for a in A:
    n=min(len(F[a]),len(ref))
    out[a]=[round(1-float(F[a][i]@ref[i]),4) for i in range(n)]
    print("%-22s per-frame: %s ..."%(a,out[a][:6]),flush=True)
json.dump(out,open(os.environ["SCRATCH"]+"/dino_curve.json","w"),indent=1); print("DONE",flush=True)
