import os
import torch, numpy as np, json, os, itertools
torch.set_num_threads(4); torch.hub.set_dir(os.environ["SCRATCH"]+"/.torch_hub")
m = torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).eval()
mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
H=os.environ["SCRATCH"]+"/lingbot-world-v2/specwm_out/hands"
A=["base_official_p1","base_p1","base_p2","fast_p1","fast_p2"]
@torch.no_grad()
def feats(V,bs=2):
    o=[]
    for i in range(0,len(V),bs):
        b=torch.from_numpy(V[i:i+bs]).permute(0,3,1,2).float()/255.
        b=torch.nn.functional.interpolate(b,size=(518,518),mode="bicubic",align_corners=False)
        o.append(torch.nn.functional.normalize(m(((b-mean)/std)).flatten(1).float(),dim=-1))
    return torch.cat(o)
F={a:feats(np.load(os.path.join(H,a,"frames_all.npy"))) for a in A}
out={}
for x,y in itertools.combinations(A,2):
    n=min(len(F[x]),len(F[y])); d=float(np.mean([1-float(p@q) for p,q in zip(F[x][:n],F[y][:n])]))
    out["%s|%s"%(x,y)]=round(d,4); print("%-22s %-22s %.4f"%(x,y,d),flush=True)
json.dump(out,open(os.environ["SCRATCH"]+"/dino_pair.json","w"),indent=1); print("DONE",flush=True)
