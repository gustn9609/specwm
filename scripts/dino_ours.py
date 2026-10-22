import os
import torch, numpy as np, json, os
torch.set_num_threads(4); torch.hub.set_dir(os.environ["SCRATCH"]+"/.torch_hub")
m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval()
mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
B = os.environ["SCRATCH"]+"/lingbot-world-v2/specwm_out/ours_same/caseB_hands"
@torch.no_grad()
def feats(V, bs=2):
    out = []
    for i in range(0, len(V), bs):
        b = torch.from_numpy(V[i:i+bs]).permute(0,3,1,2).float()/255.
        b = torch.nn.functional.interpolate(b, size=(518,518), mode="bicubic", align_corners=False)
        out.append(torch.nn.functional.normalize(m(((b-mean)/std)).flatten(1).float(), dim=-1))
    return torch.cat(out)
F = {}
for n in ["FAST_bf16", "OURS_gptq6_kv4_tiled", "SEED2_noisefloor"]:
    p = os.path.join(B, n, "frames_all.npy")
    V = np.load(p); F[n] = feats(V); print("[%s] %d frames" % (n, len(V)), flush=True)
ref = F["FAST_bf16"]; res = {}
for n in ["OURS_gptq6_kv4_tiled", "SEED2_noisefloor"]:
    d = [1 - float(a @ b) for a, b in zip(F[n], ref)]
    res[n] = dict(mean=round(float(np.mean(d)),4), max=round(float(np.max(d)),4), first=round(d[0],4), last=round(d[-1],4))
    print("%-24s vs FAST: mean %.4f  max %.4f  first %.4f  last %.4f" % (n, res[n]["mean"], res[n]["max"], res[n]["first"], res[n]["last"]), flush=True)
json.dump(res, open(os.environ["SCRATCH"]+"/dino_ours.json","w"), indent=1)
print("DONE", flush=True)
