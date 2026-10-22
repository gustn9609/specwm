import os
import torch, numpy as np, json, os, time
torch.set_num_threads(4); torch.hub.set_dir(os.environ["SCRATCH"]+"/.torch_hub")
m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval()
mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
R = os.environ["SCRATCH"]+"/lingbot-world-v2/specwm_out"
ARMS = [("upper bound (ref)", R+"/resync55/W36_sink18/resync_k6_s1234/frames_all.npy"),
        ("fast alone (W24)", R+"/resync55/W24_sink18/fast_alone_s1234/frames_all.npy"),
        ("ideal resync k6", R+"/resync55/W24_sink18/resync_k6_s1234/frames_all.npy"),
        ("control (no inject)", R+"/resync55/W24_sink18/ctrl_stale_k20_s1234/frames_all.npy"),
        ("stale d=15 s", R+"/resync55/W24_sink18/stale_d15_k20_s1234/frames_all.npy"),
        ("stale d=30 s", R+"/resync55/W24_sink18/stale_d30_k40_s1234/frames_all.npy"),
        ("stale d=45 s", R+"/resync55/W24_sink18/stale_d45_k60_s1234/frames_all.npy")]
MID = list(range(72, 111, 4)); LATE = list(range(180, 219, 2)); IDX = sorted(set(MID + LATE))

@torch.no_grad()
def feats(V, idx, bs=2):
    out = []
    for i in range(0, len(idx), bs):
        b = torch.from_numpy(V[idx[i:i+bs]]).permute(0,3,1,2).float()/255.
        b = torch.nn.functional.interpolate(b, size=(518,518), mode="bicubic", align_corners=False)
        out.append(torch.nn.functional.normalize(m(((b-mean)/std)).flatten(1).float(), dim=-1))
    return torch.cat(out)

FE = {}; t0 = time.time()
for name, p in ARMS:
    if not os.path.exists(p): print("MISSING", p, flush=True); continue
    V = np.load(p); idx = [i for i in IDX if i < len(V)]
    FE[name] = (feats(V, idx), {k: j for j, k in enumerate(idx)})
    print("[feat] %s  %.0fs" % (name, time.time()-t0), flush=True)

ref, rpos = FE["upper bound (ref)"]
res = {}
for name, (F, pos) in FE.items():
    def dist(ii):
        v = [1 - float(F[pos[i]] @ ref[rpos[i]]) for i in ii if i in pos and i in rpos]
        return round(float(np.mean(v)), 4) if v else None
    res[name] = dict(to_ref_late=dist(LATE), to_ref_mid=dist(MID))
    print("%-20s  to-ideal(45-54s) %s   to-ideal(18-27s) %s" % (name, res[name]["to_ref_late"], res[name]["to_ref_mid"]), flush=True)
json.dump(res, open(os.environ["SCRATCH"]+"/dino_stale.json","w"), indent=1)
print("DONE %.0fs" % (time.time()-t0), flush=True)
