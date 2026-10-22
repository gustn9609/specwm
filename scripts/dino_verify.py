import os
import torch, numpy as np, json
torch.set_num_threads(4); torch.hub.set_dir(os.environ["SCRATCH"]+"/.torch_hub")
m = torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).eval()
mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
R=os.environ["SCRATCH"]+"/lingbot-world-v2/specwm_out"
C=np.load(R+"/e2e55/control/frames_all.npy"); U=np.load(R+"/resync55/W36_sink18/resync_k6_s1234/frames_all.npy")
rows=json.load(open(R+"/handoff/verify/verify.json"))["rows"]
@torch.no_grad()
def f(img):
    b=torch.from_numpy(img[None]).permute(0,3,1,2).float()/255.
    b=torch.nn.functional.interpolate(b,size=(518,518),mode="bicubic",align_corners=False)
    return torch.nn.functional.normalize(m(((b-mean)/std)).flatten(1).float(),dim=-1)[0]
out=[]
for r in rows:
    k=r["chunk"]; i=max(0,(12*k-3)//4); i=min(i,len(C)-1,len(U)-1)
    d=1-float(f(C[i])@f(U[i])); out.append(dict(chunk=k,idx=i,verifier=r["dist_rel"],drift=round(d,4)))
    print("k=%2d idx=%3d verifier %.4f  drift %.4f"%(k,i,r["dist_rel"],d),flush=True)
json.dump(out,open(os.environ["SCRATCH"]+"/verify_corr.json","w"),indent=1); print("DONE",flush=True)
