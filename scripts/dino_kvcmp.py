import os
import torch, numpy as np, json, os
torch.set_num_threads(4); torch.hub.set_dir(os.environ["SCRATCH"]+"/.torch_hub")
m = torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).eval()
mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
R=os.environ["SCRATCH"]+"/lingbot-world-v2/specwm_out"; K=R+"/handoff/kvcmp/W18_sink6"
@torch.no_grad()
def feats(V,idx,bs=2):
    o=[]
    for i in range(0,len(idx),bs):
        b=torch.from_numpy(V[idx[i:i+bs]]).permute(0,3,1,2).float()/255.
        b=torch.nn.functional.interpolate(b,size=(518,518),mode="bicubic",align_corners=False)
        o.append(torch.nn.functional.normalize(m(((b-mean)/std)).flatten(1).float(),dim=-1))
    return torch.cat(o)
ref=np.load(R+"/repro/S05/base16_s4321/frames_all.npy"); n=len(ref)
idx=list(range(n))   # 4fps 저장본 전체
Fr=feats(ref,idx)
# 청크 6 이후 = 픽셀 프레임 69+ = 4fps 인덱스 18+
after=[i for i in idx if i>=18]
res={}
for arm in ["fast_alone","kvrw_refine","kvrw_ideal","kvrw_ho3","kvrw_ho3_1step","kvrw_ho2"]:
    p="%s/%s_s4321/frames_all.npy"%(K,arm)
    if not os.path.exists(p): print("MISS",arm); continue
    V=np.load(p); nn=min(len(V),n); F=feats(V,list(range(nn)))
    d_after=float(np.mean([1-float(F[i]@Fr[i]) for i in after if i<nn]))
    d_all=float(np.mean([1-float(F[i]@Fr[i]) for i in range(nn)]))
    res[arm]=dict(after6=round(d_after,4),all=round(d_all,4),n=int(nn))
    print("%-16s to base16 after chunk6: %.4f   whole: %.4f"%(arm,d_after,d_all),flush=True)
fa=res["fast_alone"]["after6"]; idl=res["kvrw_ideal"]["after6"]
print("\n회수율 = (fast_alone - arm)/(fast_alone - ideal)   [fast_alone %.4f, ideal %.4f]"%(fa,idl))
for arm in ["kvrw_refine","kvrw_ho3","kvrw_ho3_1step","kvrw_ho2"]:
    r=(fa-res[arm]["after6"])/(fa-idl+1e-9); res[arm]["retention"]=round(r,3); print("  %-16s %+.1f%%"%(arm,100*r))
json.dump(res,open(os.environ["SCRATCH"]+"/kvcmp_dino.json","w"),indent=1); print("DONE",flush=True)
