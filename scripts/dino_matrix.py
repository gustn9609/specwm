import os
import torch, numpy as np, json
torch.hub.set_dir(os.environ["SCRATCH"]+"/.torch_hub")
m=torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).eval()
R=os.environ["SCRATCH"]+"/lingbot-world-v2/specwm_out/revisit"
mean=torch.tensor([0.485,0.456,0.406]).view(3,1,1); std=torch.tensor([0.229,0.224,0.225]).view(3,1,1)
# 궤적 각도 (0->90->0->90, 161 프레임), frames_all은 4프레임 간격
angs=np.concatenate([np.linspace(0,90,54),np.linspace(90,0,54)[1:],np.linspace(0,90,54)[1:]])[:161]
out={}
for run in ["fast_yaw2","base_yaw2"]:
    V=np.load(f"{R}/{run}/frames_all.npy")
    n=len(V); idxs=np.arange(n)*4; idxs=idxs[idxs<len(angs)]; V=V[:len(idxs)]
    A=angs[idxs]
    X=torch.from_numpy(V).permute(0,3,1,2).float()/255
    X=torch.nn.functional.interpolate(X,size=(518,518),mode="bilinear"); X=(X-mean)/std
    F=[]
    with torch.no_grad():
        for i in range(0,len(X),8):
            F.append(torch.nn.functional.normalize(m(X[i:i+8]).flatten(1),dim=-1))
    F=torch.cat(F); S=(F@F.T).cpu().numpy()
    # 시간 간격별 평균, 그리고 같은각/다른각 분리
    recs=[]
    for i in range(len(A)):
        for j in range(i+1,len(A)):
            dt=(idxs[j]-idxs[i])/16.0; da=abs(A[j]-A[i])
            recs.append((dt,da,S[i,j]))
    recs=np.array(recs)
    print(f"\n=== {run}  ({len(A)} frames, {idxs[-1]} max idx) ===")
    print(f"{'gap(s)':>8}{'same-angle(<10deg)':>20}{'diff-angle(>60deg)':>20}{'delta':>9}")
    for lo,hi in [(2.5,4.5),(5.5,7.5),(8.5,10.5)]:
        sel=(recs[:,0]>=lo)&(recs[:,0]<hi)
        same=recs[sel&(recs[:,1]<10)]; diff=recs[sel&(recs[:,1]>60)]
        if len(same) and len(diff):
            print(f"{(lo+hi)/2:8.1f}{same[:,2].mean():20.4f}{diff[:,2].mean():20.4f}"
                  f"{same[:,2].mean()-diff[:,2].mean():+9.4f}   (n={len(same)},{len(diff)})")
    # 전체 회귀: 유사도 ~ 시간간격 + 각도차
    sel=recs[:,0]>=2.0
    r=recs[sel]
    import numpy.linalg as la
    Xd=np.c_[np.ones(len(r)), r[:,0], r[:,1]]
    b=la.lstsq(Xd,r[:,2],rcond=None)[0]
    print(f"  회귀: sim = {b[0]:.3f} {b[1]:+.4f}*gap(s) {b[2]:+.5f}*angle_diff(deg)")
    print(f"        -> 각도차 90도의 효과 {b[2]*90:+.4f} / 시간 6.7초의 효과 {b[1]*6.7:+.4f}")
    out[run]=dict(coef_gap=float(b[1]),coef_angle=float(b[2]),
                  revisit=float(S[list(idxs).index(52) if 52 in idxs else 13, -1]))
json.dump(out,open(f"{R}/dino_matrix.json","w"),indent=1)
