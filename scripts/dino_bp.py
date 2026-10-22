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
        b=torch.from_numpy(V[i:i+bs]).permute(0,3,1,2).float()/255.; b=torch.nn.functional.interpolate(b,size=(518,518),mode="bicubic",align_corners=False)
        o.append(torch.nn.functional.normalize(m(((b-mean)/std)).flatten(1).float(),dim=-1))
    return torch.cat(o)
ref=feats(np.load(R+"/revisit/base_official_S05_s16/frames_all.npy")[:39]); ref70=feats(np.load(R+"/revisit/base_official_S05/frames_all.npy")[:39])
out={}
for S in ["sched40","sched60"]:
    for a in ["fast_alone","hyb_t010","hyb_t020","hyb_t040","base_refined"]:
        F=feats(np.load(R+"/basepatch/%s/%s/frames_all.npy"%(S,a))); n=min(len(F),39)
        d16=float(np.mean([1-float(F[i]@ref[i]) for i in range(n)])); d70=float(np.mean([1-float(F[i]@ref70[i]) for i in range(n)]))
        out["%s/%s"%(S,a)]=dict(to_base16=round(d16,4),to_base70=round(d70,4)); print("%-22s to Base16 %.4f  to Base70 %.4f"%("%s/%s"%(S,a),d16,d70),flush=True)
# LISA 확신도 arm
for a in ["conf_t002_k005","conf_t002_k010","conf_t005_k005","conf_t005_k010"]:
    try:
        d=json.load(open(R+"/patchspec/%s/patch_spec.json"%a)); k=d["keep_frac"]; print("%-16s keep node1/2/3 = %s"%(a,[k[n] for n in "012"]),flush=True)
    except Exception as e: print(a,"ERR",e)
json.dump(out,open(os.environ["SCRATCH"]+"/dino_bp.json","w"),indent=1); print("DONE")
