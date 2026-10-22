import os
"""resync 결과 분석 (서버 gh-dev):
  drift-to-teacher(t): 각 arm 프레임과 Base 프레임(같은 인덱스)의 DINO cos 거리, 청크별
  snap: 재접지 경계(k*12 프레임)에서 |frame[t]-frame[t-1]| 픽셀 점프 / 그 arm 의 중앙값 인접 차
  VBench 는 별도 잡.
"""
import numpy as np, torch, os, json, glob
S = os.environ["SCRATCH"]+""; R = S + "/lingbot-world-v2/specwm_out/resync_s05"
torch.hub.set_dir(S + "/.torch_hub")
m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval().cuda()
mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1).cuda(); std = torch.tensor([0.229,0.224,0.225]).view(3,1,1).cuda()
def feats(V):
    X = torch.from_numpy(V).permute(0,3,1,2).float().cuda()/255
    X = torch.nn.functional.interpolate(X, size=(518,518), mode="bilinear"); X = (X-mean)/std
    o = []
    with torch.no_grad():
        for i in range(0, len(X), 8): o.append(torch.nn.functional.normalize(m(X[i:i+8]).flatten(1), dim=-1).cpu())
    return torch.cat(o)
B = np.load(R + "/base_frames.npy"); FB = feats(B)          # 39 프레임 (4간격) = 청크당 3
out = {}
for d in sorted(glob.glob(R + "/*/arm.json")):
    tag = os.path.basename(os.path.dirname(d)); V = np.load(os.path.dirname(d) + "/frames_all.npy")
    n = min(len(V), len(B)); F = feats(V[:n])
    dist = 1 - (F * FB[:n]).sum(-1).numpy()                   # 프레임별 DINO 거리 (Base 대비)
    per_chunk = [float(dist[c*3:(c+1)*3].mean()) for c in range(n // 3)]
    v = V[:n].astype(np.float32)/255; dif = np.abs(v[1:]-v[:-1]).mean(axis=(1,2,3))
    plan = json.load(open(d))["plan"]
    snaps = {}
    for k in (3, 6, 9):
        if k*3 < len(dif): snaps["k%d" % k] = float(dif[k*3-1] / (np.median(dif) + 1e-9))   # 4프레임 간격 저장이라 청크 경계 = 3k
    out[tag] = dict(plan=plan, dist_per_chunk=per_chunk, dist_mean_after6=float(np.mean(per_chunk[6:])), snap_ratio=snaps, flicker=float(dif.mean()))
    print("  %-20s plan=%s  dist(0-5)=%.3f dist(6-12)=%.3f  snap=%s" % (tag, plan, np.mean(per_chunk[:6]), np.mean(per_chunk[6:]), {k: round(v,2) for k,v in snaps.items()}), flush=True)
json.dump(out, open(R + "/resync_analysis.json", "w"), indent=1); print("[RESYNC_ANALYSIS_DONE]", flush=True)
