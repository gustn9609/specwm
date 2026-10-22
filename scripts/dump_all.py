import os
"""모든 신규 arm 의 (시간간격, 자세각차, DINO유사도) 페어 저장."""
import torch, numpy as np, os, glob, json
torch.hub.set_dir(os.environ["SCRATCH"]+"/.torch_hub")
S = os.environ["SCRATCH"]+""
R = S + "/lingbot-world-v2/specwm_out"
IDX = {"00": np.load(S + "/rt2_idx.npy"),
       "01": np.load(S + "/revisit_rt_idx_01.npy"),
       "05": np.load(S + "/revisit_rt_idx_05.npy")}
POS = {k: np.load(S + "/poses_%s.npy" % k) for k in ["00", "01", "05"]}
FWD = {k: -v[:, :3, :3][:, :, 2] for k, v in POS.items()}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval().to(DEV)
print("device:", DEV, flush=True)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEV)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEV)

def feats(V):
    X = torch.from_numpy(V).permute(0, 3, 1, 2).float().to(DEV) / 255
    X = torch.nn.functional.interpolate(X, size=(518, 518), mode="bilinear")
    X = (X - mean) / std
    o = []
    with torch.no_grad():
        for i in range(0, len(X), 4):
            o.append(torch.nn.functional.normalize(m(X[i:i+4]).flatten(1), dim=-1).cpu())
    return torch.cat(o)

def pairs(path, scene):
    V = np.load(path); idx = IDX[scene]; fwd = FWD[scene]
    fi = np.arange(len(V)) * 4; fi = fi[fi < len(idx)]; V = V[:len(fi)]
    F = feats(V); Sm = (F @ F.T).numpy(); rec = []
    for i in range(len(fi)):
        for j in range(i + 1, len(fi)):
            da = np.degrees(np.arccos(np.clip((fwd[idx[fi[i]]] * fwd[idx[fi[j]]]).sum(), -1, 1)))
            rec.append(((fi[j] - fi[i]) / 16.0, da, Sm[i, j]))
    return np.array(rec)

out = {}
# 다장면
for d in sorted(glob.glob(R + "/multi_scene/S*")):
    tag = os.path.basename(d); scene = tag[1:3]
    p = os.path.join(d, "frames_all.npy")
    if not os.path.exists(p): continue
    out["ms_" + tag] = pairs(p, scene); print("  ms_%-24s %d pairs" % (tag, len(out["ms_" + tag])), flush=True)
# 확장 창 스윕 (전부 장면 00)
for d in sorted(glob.glob(R + "/win_sweep2/W*")):
    tag = os.path.basename(d)
    p = os.path.join(d, "frames_all.npy")
    if not os.path.exists(p): continue
    out["w2_" + tag] = pairs(p, "00"); print("  w2_%-24s %d pairs" % (tag, len(out["w2_" + tag])), flush=True)
np.savez(S + "/all_pairs.npz", **out)
print("저장: %d arm" % len(out), flush=True)
