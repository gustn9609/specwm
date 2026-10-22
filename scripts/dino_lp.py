import os
import torch, numpy as np, json, os, time
torch.set_num_threads(4); torch.hub.set_dir(os.environ["SCRATCH"]+"/.torch_hub")
m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval()
mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
R = os.environ["SCRATCH"]+"/lingbot-world-v2/specwm_out"
ARMS = [
 ("no correction",      R+"/e2e55/control/frames_all.npy"),
 ("delta = 0 s",        R+"/lat55/d0/frames_all.npy"),
 ("delta = 0.75 s",     R+"/lat55/d1/frames_all.npy"),
 ("delta = 2.25 s",     R+"/lat55/d3/frames_all.npy"),
 ("delta = 5.25 s",     R+"/lat55/d7/frames_all.npy"),
 ("every 30 s (2x)",    R+"/per55/every30/frames_all.npy"),
 ("every 15 s (4x)",    R+"/per55/every15/frames_all.npy"),
 ("3x fixed",           R+"/e2e55/fixed/frames_all.npy"),
 ("3x rolling",         R+"/e2e55/rolling/frames_all.npy"),
 ("upper bound",        R+"/resync55/W36_sink18/resync_k6/frames_all.npy"),
]
ANCH = list(range(0, 18, 2)); MID = list(range(72, 111, 4)); LATE = list(range(180, 219, 2))

@torch.no_grad()
def feats(V, idx, bs=2):
    out = []
    for i in range(0, len(idx), bs):
        b = torch.from_numpy(V[idx[i:i+bs]]).permute(0,3,1,2).float()/255.
        b = torch.nn.functional.interpolate(b, size=(518,518), mode="bicubic", align_corners=False)
        out.append(torch.nn.functional.normalize(m(((b-mean)/std)).flatten(1).float(), dim=-1))
    return torch.cat(out)

res = {}; t0 = time.time()
for name, p in ARMS:
    if not os.path.exists(p): print("MISSING", p, flush=True); continue
    V = np.load(p)
    idx = sorted(set(ANCH + MID + LATE + list(range(0, len(V), 12))))
    idx = [i for i in idx if i < len(V)]
    F = feats(V, idx); pos = {k: j for j, k in enumerate(idx)}
    anc = torch.nn.functional.normalize(F[[pos[i] for i in ANCH if i in pos]].mean(0), dim=-1)
    drift = lambda ii: float(np.mean([1 - float(F[pos[i]] @ anc) for i in ii if i in pos]))
    def motion(ii):
        v = [1 - float(F[pos[i]] @ F[pos[i-4]]) for i in ii if i in pos and (i-4) in pos]
        return float(np.mean(v)) if v else float("nan")
    v = V.astype(np.float32)[180:]
    res[name] = dict(drift_late=round(drift(LATE),4), drift_mid=round(drift(MID),4),
                     motion_late=round(motion(LATE),4), motion_mid=round(motion(MID),4),
                     magenta=round(float(((v[...,0]+v[...,2])/2 - v[...,1]).mean()),2),
                     bright=round(float(v.mean()),1), nframes=int(len(V)))
    r = res[name]
    print("%-18s drift45-54 %.4f  drift18-27 %.4f  motion45-54 %.4f  magenta %6.2f" %
          (name, r["drift_late"], r["drift_mid"], r["motion_late"], r["magenta"]), flush=True)
json.dump(res, open(os.environ["SCRATCH"]+"/dino_lat_per.json","w"), indent=1)
print("DONE %.0fs" % (time.time()-t0), flush=True)
