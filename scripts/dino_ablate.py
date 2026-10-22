import os
import torch, numpy as np, json, os
torch.set_num_threads(4); torch.hub.set_dir(os.environ["SCRATCH"]+"/.torch_hub")
m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval()
mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
H = os.environ["SCRATCH"]+"/lingbot-world-v2/specwm_out/hands"
ARMS = ["base_official_p1","base_official_p1_s16","base_official_p1_s4",
        "base_causal_N12","base_causal_N6","base_causal_N3","fast_p1",
        "base_p1","base_p2","fast_p2"]
@torch.no_grad()
def feats(V, bs=2):
    out=[]
    for i in range(0,len(V),bs):
        b = torch.from_numpy(V[i:i+bs]).permute(0,3,1,2).float()/255.
        b = torch.nn.functional.interpolate(b, size=(518,518), mode="bicubic", align_corners=False)
        out.append(torch.nn.functional.normalize(m(((b-mean)/std)).flatten(1).float(), dim=-1))
    return torch.cat(out)
F={}
for a in ARMS:
    p = os.path.join(H,a,"frames_all.npy")
    if not os.path.exists(p): print("MISS",a,flush=True); continue
    V = np.load(p); F[a]=feats(V); print("[%s] %d frames"%(a,len(V)),flush=True)
ref = F["base_official_p1"]
res={}
for a,f in F.items():
    n=min(len(f),len(ref))
    d=[1-float(x@y) for x,y in zip(f[:n],ref[:n])]
    res[a]=dict(to_base70=round(float(np.mean(d)),4), last=round(float(np.mean(d[-5:])),4))
    print("%-24s to Base70: mean %.4f  last5 %.4f"%(a,res[a]["to_base70"],res[a]["last"]),flush=True)
json.dump(res, open(os.environ["SCRATCH"]+"/dino_ablate.json","w"), indent=1)
print("DONE",flush=True)
