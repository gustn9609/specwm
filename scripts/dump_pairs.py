import os
"""프레임쌍 (시간간격, 자세각차, 유사도) 원본을 저장 -> 이후 로컬 분석용"""
import torch, numpy as np, sys
torch.hub.set_dir(os.environ["SCRATCH"]+"/.torch_hub")
R=os.environ["SCRATCH"]+"/lingbot-world-v2/specwm_out/revisit"
IDX=np.load(os.environ["SCRATCH"]+"/rt2_idx.npy"); P=np.load(os.environ["SCRATCH"]+"/poses_00.npy")
FWD=-P[:,:3,:3][:,:,2]
m=torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).eval()
mean=torch.tensor([0.485,0.456,0.406]).view(3,1,1); std=torch.tensor([0.229,0.224,0.225]).view(3,1,1)
def feats(V):
    X=torch.from_numpy(V).permute(0,3,1,2).float()/255
    X=torch.nn.functional.interpolate(X,size=(518,518),mode="bilinear"); X=(X-mean)/std
    o=[]
    with torch.no_grad():
        for i in range(0,len(X),4): o.append(torch.nn.functional.normalize(m(X[i:i+4]).flatten(1),dim=-1))
    return torch.cat(o)
out={}
for run in sys.argv[1:]:
    V=np.load(f"{R}/{run}/frames_all.npy"); fi=np.arange(len(V))*4; fi=fi[fi<len(IDX)]; V=V[:len(fi)]
    S=(feats(V)@feats(V).T).numpy(); rec=[]
    for i in range(len(fi)):
        for j in range(i+1,len(fi)):
            da=np.degrees(np.arccos(np.clip((FWD[IDX[fi[i]]]*FWD[IDX[fi[j]]]).sum(),-1,1)))
            rec.append(((fi[j]-fi[i])/16.0, da, S[i,j]))
    out[run]=np.array(rec); print(run, out[run].shape, flush=True)
np.savez(os.environ["SCRATCH"]+"/rt2_pairs.npz", **out); print("saved")
