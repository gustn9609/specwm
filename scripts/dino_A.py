import os
import torch, numpy as np, json
torch.set_num_threads(4); torch.hub.set_dir(os.environ["SCRATCH"]+"/.torch_hub")
m = torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).eval()
mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
R=os.environ["SCRATCH"]+"/lingbot-world-v2/specwm_out"
@torch.no_grad()
def feats(V,bs=2):
    o=[]
    for i in range(0,len(V),bs):
        b=torch.from_numpy(V[i:i+bs]).permute(0,3,1,2).float()/255.
        b=torch.nn.functional.interpolate(b,size=(518,518),mode="bicubic",align_corners=False)
        o.append(torch.nn.functional.normalize(m(((b-mean)/std)).flatten(1).float(),dim=-1))
    return torch.cat(o)
ref=feats(np.load(R+"/revisit/base_official_S05_s16/frames_all.npy"))
S=feats(np.load(R+"/ideasAB/A_sequential/frames_all.npy")[::4]); P=feats(np.load(R+"/ideasAB/A_pipeline/frames_all.npy")[::4])
n=min(len(S),len(P),len(ref))
d=lambda A,B,ii:float(np.mean([1-float(A[i]@B[i]) for i in ii]))
out=dict(seq_to_base16=round(d(S,ref,range(n)),4),pipe_to_base16=round(d(P,ref,range(n)),4),seq_vs_pipe=round(d(S,P,range(n)),4),
         seq_last5=round(d(S,ref,range(n-5,n)),4),pipe_last5=round(d(P,ref,range(n-5,n)),4))
print(json.dumps(out)); json.dump(out,open(os.environ["SCRATCH"]+"/A_dino.json","w"),indent=1)
