import os
"""창 스윕 + Case A 각 arm의 (시간간격, 자세각차, DINO유사도) 페어를 저장."""
import torch, numpy as np, os, sys
torch.hub.set_dir(os.environ["SCRATCH"]+"/.torch_hub")
IDX = np.load(os.environ["SCRATCH"]+"/rt2_idx.npy")
P = np.load(os.environ["SCRATCH"]+"/poses_00.npy")
FWD = -P[:, :3, :3][:, :, 2]
m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval()
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def feats(V):
    X = torch.from_numpy(V).permute(0, 3, 1, 2).float() / 255
    X = torch.nn.functional.interpolate(X, size=(518, 518), mode="bilinear")
    X = (X - mean) / std
    o = []
    with torch.no_grad():
        for i in range(0, len(X), 4):
            o.append(torch.nn.functional.normalize(m(X[i:i+4]).flatten(1), dim=-1))
    return torch.cat(o)

ROOT = os.environ["SCRATCH"]+"/lingbot-world-v2/specwm_out"
SRC = [("win_%s" % a, "%s/win_sweep/%s/frames_all.npy" % (ROOT, a))
       for a in ["W18_bf16", "W18_kv4", "W36_kv4", "W18_sink0"]] + \
      [("caseA_%s" % a, "%s/ours_same/caseA_revisit_fixed/%s/frames_all.npy" % (ROOT, a))
       for a in ["FAST_bf16", "SEED2_noisefloor", "OURS_gptq6_kv4_tiled"]]
out = {}
for tag, path in SRC:
    if not os.path.exists(path):
        print("  없음:", tag, path, flush=True); continue
    V = np.load(path)
    fi = np.arange(len(V)) * 4
    fi = fi[fi < len(IDX)]; V = V[:len(fi)]
    F = feats(V); S = (F @ F.T).numpy()
    rec = []
    for i in range(len(fi)):
        for j in range(i + 1, len(fi)):
            da = np.degrees(np.arccos(np.clip((FWD[IDX[fi[i]]] * FWD[IDX[fi[j]]]).sum(), -1, 1)))
            rec.append(((fi[j] - fi[i]) / 16.0, da, S[i, j]))
    out[tag] = np.array(rec)
    print("  %-24s %d frames, %d pairs" % (tag, len(fi), len(rec)), flush=True)
np.savez(os.environ["SCRATCH"]+"/win_pairs.npz", **out)
print("저장 완료", flush=True)
