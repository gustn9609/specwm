import os
"""참조 없는 품질 지표 2종 (55초 40개 실행).
(1) patch-FD : 프레임을 64x64 패치로 잘라 DINOv2 특징 -> Base70(10초, 41프레임) 패치 분포와의 Frechet 거리. 시각(t)에 무관하게 '질감/내용 통계' 만 비교.
(2) liveness : 1초 간격 DINO 거리의 평균 (얼어붙음 탐지) + 자홍 편향.
6초 창(24프레임@4fps) 단위로 내서 시간 곡선도 남긴다.
"""
import torch, numpy as np, json, os, sys
torch.set_num_threads(4); torch.hub.set_dir(os.environ["SCRATCH"]+"/.torch_hub")
m = torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).eval()
mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
R=os.environ["SCRATCH"]+"/lingbot-world-v2/specwm_out"
RUNS = {
 "e2e55/control":0,"e2e55/fixed":0,"e2e55/rolling":0,"e2e55_S01/fixed":1,"e2e55_S01/rolling":1,"e2e55_S02/fixed":2,"e2e55_S02/rolling":2,
 "lat55/d0":0,"lat55/d1":0,"lat55/d3":0,"lat55/d7":0,"lat55_s4321/control":0,"lat55_s4321/d0":0,"lat55_s4321/d3":0,"lat55_s4321/d7":0,
 "lat55_s9999/control":0,"lat55_s9999/d0":0,"lat55_s9999/d3":0,"lat55_s9999/d7":0,"per55/every15":0,"per55/every30":0,"per55_s4321/every30":0,"per55_s9999/every30":0,
 "resync55/W36_sink18/fast_alone_s1234":0,"resync55/W36_sink18/resync_k6_s1234":0,"resync55/W36_sink18/kvrw_naive_s1234":0,
 "resync55/W24_sink18/stale_d15_k20_s1234":0,"resync55/W24_sink18/stale_d30_k40_s1234":0,"resync55/W24_sink18/stale_d45_k60_s1234":0,"resync55/W24_sink18/ctrl_stale_k20_s1234":0,
}
REF = {0:R+"/revisit/base_official_S05/frames_all.npy",1:R+"/revisit/base_official_S01/frames_all.npy",2:R+"/revisit/base_official_S02/frames_all.npy"}
P=64; STRIDE=64
@torch.no_grad()
def patch_feats(V, idx, per_frame=24):
    out=[]
    for i in idx:
        img=V[i]; H,W,_=img.shape; ys=range(0,H-P+1,STRIDE); xs=range(0,W-P+1,STRIDE)
        pts=[img[y:y+P,x:x+P] for y in ys for x in xs]
        rng=np.random.RandomState(i); sel=rng.choice(len(pts),min(per_frame,len(pts)),replace=False)
        b=torch.from_numpy(np.stack([pts[j] for j in sel])).permute(0,3,1,2).float()/255.
        b=torch.nn.functional.interpolate(b,size=(112,112),mode="bicubic",align_corners=False)
        out.append(m(((b-mean)/std)).float())
    return torch.cat(out).numpy()
def frechet(A,B):
    mu1,mu2=A.mean(0),B.mean(0); C1,C2=np.cov(A,rowvar=False),np.cov(B,rowvar=False)
    from scipy.linalg import sqrtm
    cs=sqrtm(C1@C2); cs=cs.real if np.iscomplexobj(cs) else cs
    return float(((mu1-mu2)**2).sum()+np.trace(C1+C2-2*cs))
@torch.no_grad()
def frame_feats(V, idx, bs=2):
    o=[]
    for i in range(0,len(idx),bs):
        b=torch.from_numpy(V[idx[i:i+bs]]).permute(0,3,1,2).float()/255.
        b=torch.nn.functional.interpolate(b,size=(518,518),mode="bicubic",align_corners=False)
        o.append(torch.nn.functional.normalize(m(((b-mean)/std)).flatten(1).float(),dim=-1))
    return torch.cat(o)
refF={}; refFull={}
for sc,p in REF.items():
    if not os.path.exists(p): print("MISS ref",p,flush=True); continue
    Vr=np.load(p); refF[sc]=patch_feats(Vr,list(range(0,len(Vr),2))); print("[ref S0%d] %d patches"%(sc,len(refF[sc])),flush=True)
    # 참조 자체의 잡음: 앞 절반 vs 뒤 절반
    h=len(refF[sc])//2; print("   ref self-FD (first half vs second half): %.2f"%frechet(refF[sc][:h],refF[sc][h:]),flush=True)
res={}
for tag,sc in RUNS.items():
    p="%s/%s/frames_all.npy"%(R,tag)
    if not os.path.exists(p) or sc not in refF: print("MISS",tag,flush=True); continue
    V=np.load(p); n=len(V); win=24
    fd_w=[]; live_w=[]; mag_w=[]
    for w0 in range(0,n-win+1,win):
        idx=list(range(w0,w0+win,2)); Fp=patch_feats(V,idx,per_frame=24); fd_w.append(frechet(Fp,refF[sc]))
        Ff=frame_feats(V,list(range(w0,w0+win,4))); live_w.append(float(np.mean([1-float(Ff[j]@Ff[j-1]) for j in range(1,len(Ff))])))
        v=V[w0:w0+win].astype(np.float32); mag_w.append(float(((v[...,0]+v[...,2])/2-v[...,1]).mean()))
    res[tag]=dict(scene=sc,fd_windows=[round(x,2) for x in fd_w],fd_mean=round(float(np.mean(fd_w)),2),fd_last=round(fd_w[-1],2),fd_first=round(fd_w[0],2),
                  live_windows=[round(x,4) for x in live_w],live_mean=round(float(np.mean(live_w)),4),magenta_last=round(mag_w[-1],2))
    print("%-42s FD first %6.1f  mean %6.1f  last %6.1f | live %.3f | magenta(last) %+.1f"%(tag,fd_w[0],np.mean(fd_w),fd_w[-1],np.mean(live_w),mag_w[-1]),flush=True)
json.dump(res,open(os.environ["SCRATCH"]+"/patch_stats.json","w"),indent=1); print("DONE",flush=True)
