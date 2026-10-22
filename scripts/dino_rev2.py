import os
import torch, numpy as np, json, os
from PIL import Image
torch.set_num_threads(4); torch.hub.set_dir(os.environ["SCRATCH"]+"/.torch_hub")
m = torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).eval()
mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
R=os.environ["SCRATCH"]+"/lingbot-world-v2/specwm_out/revisit2"
IN=os.environ["SCRATCH"]+"/lingbot-world-v2/examples/00/image.jpg"
ARMS=[("base16",1234,R+"/base16_s1234"),("base16",4321,R+"/base16_s4321"),
      ("base4",1234,R+"/base4_s1234"),("base4",4321,R+"/base4_s4321"),
      ("causalN3",1234,R+"/causalN3_s1234"),("causalN3",4321,R+"/causalN3_s4321"),
      ("fast",1234,R+"/fast/W18_sink6/fast_alone_s1234"),("fast",4321,R+"/fast/W18_sink6/fast_alone_s4321")]
@torch.no_grad()
def farr(V,bs=2):
    o=[]
    for i in range(0,len(V),bs):
        b=torch.from_numpy(V[i:i+bs]).permute(0,3,1,2).float()/255.
        b=torch.nn.functional.interpolate(b,size=(518,518),mode="bicubic",align_corners=False)
        o.append(torch.nn.functional.normalize(m(((b-mean)/std)).flatten(1).float(),dim=-1))
    return torch.cat(o)
@torch.no_grad()
def fimg(p):
    x=torch.from_numpy(np.asarray(Image.open(p).convert("RGB"))).permute(2,0,1)[None].float()/255.
    x=torch.nn.functional.interpolate(x,size=(518,518),mode="bicubic",align_corners=False)
    return torch.nn.functional.normalize(m((x-mean)/std).flatten(1).float(),dim=-1)[0]
ref=fimg(IN); res={}; curves={}
N=39                                # 공통 인덱스 0..38  (픽셀 프레임 0..152)
for tag,sd,d in ARMS:
    p=os.path.join(d,"frames_all.npy")
    if not os.path.exists(p): print("MISS",p,flush=True); continue
    V=np.load(p)[:N]; F=farr(V)
    cur=[1-float(F[i]@ref) for i in range(len(F))]
    curves["%s_s%d"%(tag,sd)]=[round(c,4) for c in cur]
    first,mid,last=cur[0],cur[20],cur[38]
    res["%s_s%d"%(tag,sd)]=dict(arm=tag,seed=sd,first=round(first,4),mid=round(mid,4),last=round(last,4),
                                gain=round(mid-last,4),peak=round(max(cur),4))
    print("%-14s s%-5d first %.4f  mid(90deg) %.4f  last %.4f  gain %+.4f  peak %.4f"%(tag,sd,first,mid,last,mid-last,max(cur)),flush=True)
json.dump({"per_arm":res,"curves":curves},open(os.environ["SCRATCH"]+"/dino_rev2.json","w"),indent=1)
print("DONE",flush=True)
