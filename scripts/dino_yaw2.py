import os
import torch, numpy as np, os, json
torch.hub.set_dir(os.environ["SCRATCH"]+"/.torch_hub")
m=torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).eval()
R=os.environ["SCRATCH"]+"/lingbot-world-v2/specwm_out/revisit"
mean=torch.tensor([0.485,0.456,0.406]).view(3,1,1); std=torch.tensor([0.229,0.224,0.225]).view(3,1,1)
def featarr(a):  # a: HxWx3 uint8
    x=torch.from_numpy(a).permute(2,0,1).float()/255
    x=torch.nn.functional.interpolate(x[None],size=(518,518),mode="bilinear")[0]
    x=(x-mean)/std
    with torch.no_grad(): return torch.nn.functional.normalize(m(x[None]).flatten(1),dim=-1)[0]
cos=lambda a,b: float((a*b).sum())
for run in ["fast_yaw2","base_yaw2"]:
    p=f"{R}/{run}/frames_all.npy"
    if not os.path.exists(p): print(f"{run}: 아직 없음"); continue
    V=np.load(p)  # 4프레임 간격
    n=len(V); i53=53//4; i160=min(160//4,n-1); i107=107//4; i0=0
    F={k:featarr(V[k]) for k in {i0,i53,i107,i160}}
    print(f"\n[{run}] frames_all {V.shape}")
    print(f"  90deg 1차(53) <-> 90deg 2차(160)  = {cos(F[i53],F[i160]):.4f}   <- 환각 장면 재방문")
    print(f"  90deg 1차(53) <-> 0deg 복귀(107)  = {cos(F[i53],F[i107]):.4f}   <- 통제군(다른 각도)")
    print(f"  입력(0)       <-> 0deg 복귀(107)  = {cos(F[i0],F[i107]):.4f}   <- 이미지 조건 복귀")
    print(f"  입력(0)       <-> 90deg 1차(53)   = {cos(F[i0],F[i53]):.4f}")
