import os
"""시간 간격을 통제한 pose 추적력 분석.

핵심 질문: Fast의 기억창(4.5초)을 넘어선 간격에서도 카메라 자세에 따라
장면이 일관되게 재현되는가? 양방향(Base)은 되고 few-step AR(Fast)은
안 된다면 그게 우리가 찾던 failure mode다.
"""
import torch, numpy as np, json, os, sys
torch.hub.set_dir(os.environ["SCRATCH"]+"/.torch_hub")
R_ = os.environ["SCRATCH"]+"/lingbot-world-v2/specwm_out/revisit"
IDX = np.load(os.environ["SCRATCH"]+"/rt2_idx.npy")       # 프레임별 원본 pose 인덱스
POSES = np.load(os.environ["SCRATCH"]+"/poses_00.npy")
FWD = -POSES[:, :3, :3][:, :, 2]                          # 카메라 시선 방향
FPS, WINDOW_S = 16.0, 4.5

m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval()
mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1); std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

def feats(V):
    X = torch.from_numpy(V).permute(0,3,1,2).float()/255
    X = torch.nn.functional.interpolate(X, size=(518,518), mode="bilinear"); X = (X-mean)/std
    out=[]
    with torch.no_grad():
        for i in range(0, len(X), 4):
            out.append(torch.nn.functional.normalize(m(X[i:i+4]).flatten(1), dim=-1))
    return torch.cat(out)

def viewang(a, b):
    return np.degrees(np.arccos(np.clip((FWD[a]*FWD[b]).sum(), -1, 1)))

res = {}
for run in sys.argv[1:]:
    V = np.load(f"{R_}/{run}/frames_all.npy")
    fi = np.arange(len(V))*4
    fi = fi[fi < len(IDX)]; V = V[:len(fi)]
    S = (feats(V) @ feats(V).T).numpy()
    rec = []
    for i in range(len(fi)):
        for j in range(i+1, len(fi)):
            dt = (fi[j]-fi[i])/FPS
            da = viewang(IDX[fi[i]], IDX[fi[j]])
            rec.append((dt, da, S[i,j]))
    rec = np.array(rec)
    print(f"\n=== {run} ===  {len(fi)} frames, {len(rec)} pairs")
    print(f"{'간격(초)':>10}{'n':>6}{'자세효과(도당)':>16}{'표준오차':>10}{'90도환산':>10}")
    bins = [(0,2),(2,3.5),(3.5,4.5),(4.5,6),(6,9.5)]
    prof = []
    for lo,hi in bins:
        sel = rec[(rec[:,0]>=lo)&(rec[:,0]<hi)]
        if len(sel) < 12 or sel[:,1].std() < 5: continue
        X = np.c_[np.ones(len(sel)), sel[:,1]]
        beta, *_ = np.linalg.lstsq(X, sel[:,2], rcond=None)
        resid = sel[:,2] - X@beta
        se = np.sqrt((resid@resid)/(len(sel)-2) * np.linalg.inv(X.T@X)[1,1])
        tag = "창안" if hi <= WINDOW_S else ("창밖" if lo >= WINDOW_S else "경계")
        print(f"{lo:5.1f}-{hi:<4.1f}{len(sel):6d}{beta[1]:16.5f}{se:10.5f}"
              f"{beta[1]*90:10.4f}   {tag}")
        prof.append(dict(lo=lo, hi=hi, n=int(len(sel)), slope=float(beta[1]),
                         se=float(se), t=float(beta[1]/se)))
    inw  = [p for p in prof if p["hi"] <= WINDOW_S]
    outw = [p for p in prof if p["lo"] >= WINDOW_S]
    f_in  = np.mean([p["slope"] for p in inw])  if inw  else float("nan")
    f_out = np.mean([p["slope"] for p in outw]) if outw else float("nan")
    print(f"  창 안쪽 평균 기울기 {f_in:+.5f} / 창 바깥 {f_out:+.5f}"
          f"  -> 유지율 {f_out/f_in*100 if f_in else float('nan'):.0f}%")
    res[run] = dict(profile=prof, in_window=f_in, out_window=f_out,
                    retention=float(f_out/f_in) if f_in else None)
json.dump(res, open(f"{R_}/rt2_analysis.json","w"), indent=1)
print("\n판정 기준: 자세효과 기울기가 음수(각도 벌어지면 덜 비슷)여야 정상 추적.")
print("Base는 창밖에서도 기울기 유지, Fast는 창밖에서 0으로 붕괴 -> failure mode 확인")
