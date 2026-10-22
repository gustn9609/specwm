import os
import torch, numpy as np, json, os
torch.set_num_threads(4); torch.hub.set_dir(os.environ["SCRATCH"]+"/.torch_hub")
m = torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).eval()
mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
R=os.environ["SCRATCH"]+"/lingbot-world-v2/specwm_out"
@torch.no_grad()
def feats(V,bs=2):
    o=[]
    for i in range(0,len(V),bs):
        b=torch.from_numpy(V[i:i+bs]).permute(0,3,1,2).float()/255.; b=torch.nn.functional.interpolate(b,size=(518,518),mode="bicubic",align_corners=False)
        o.append(torch.nn.functional.normalize(m(((b-mean)/std)).flatten(1).float(),dim=-1))
    return torch.cat(o)
ref=feats(np.load(R+"/revisit/base_official_S05_s16/frames_all.npy")); T=feats(np.load(R+"/patchspec/all_target/frames_all.npy")); D=feats(np.load(R+"/patchspec/all_draft/frames_all.npy"))
out={}
for a in ["all_draft","patch_t001","patch_t002","patch_t005","patch_t010","all_target"]:
    F=feats(np.load(R+"/patchspec/%s/frames_all.npy"%a)); n=min(len(F),len(ref),len(T))
    out[a]=dict(to_base16=round(float(np.mean([1-float(F[i]@ref[i]) for i in range(n)])),4), to_target=round(float(np.mean([1-float(F[i]@T[i]) for i in range(n)])),4), to_draft=round(float(np.mean([1-float(F[i]@D[i]) for i in range(n)])),4))
    print("%-12s to Base16 %.4f | to all_target %.4f | to all_draft %.4f"%(a,out[a]["to_base16"],out[a]["to_target"],out[a]["to_draft"]),flush=True)
json.dump(out,open(os.environ["SCRATCH"]+"/dino_patch.json","w"),indent=1)
