"""Base-vs-Base DINO distance (the floor for any 'distance to Base' drift criterion): for each task, per-chunk mean 1-cos between Base 70-step
rollouts of different seeds, averaged over chunks. usage: base_selfdist.py TASK SEED1,SEED2,... [more TASK SEEDS pairs]"""
import sys, os, json, numpy as np, torch, imageio, itertools
S = "/home/hk28965/specwm"; V0 = S + "/lingbot-world-v2/specwm_out/spec_v0"; dev = torch.device("cuda:0")
torch.hub.set_dir(S + "/.torch_hub"); dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval().to(dev)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
@torch.no_grad()
def feats(V):
    out = []
    for i in range(0, len(V), 16):
        x = torch.from_numpy(np.stack(V[i:i + 16])).to(dev).permute(0, 3, 1, 2).float() / 255; x = torch.nn.functional.interpolate(x, size=(224, 392), mode="bilinear", antialias=True)
        out.append(torch.nn.functional.normalize(dino((x - MEAN) / STD).float(), dim=-1).cpu())
    return torch.cat(out)
args = sys.argv[1:]; res = {}
for task, seeds in zip(args[::2], args[1::2]):
    F = {}
    for sd in seeds.split(","):
        p = "%s/%s/base70_s%s/video.mp4" % (V0, task, sd)
        if os.path.exists(p): F[sd] = feats([np.asarray(f) for f in imageio.get_reader(p)])
    for a, b in itertools.combinations(sorted(F), 2):
        n = min(len(F[a]), len(F[b])); d = 1 - (F[a][:n] * F[b][:n]).sum(-1)   # per frame
        ch = [float(d[i:i + 12].mean()) for i in range(0, n, 12)]; res["%s %s-%s" % (task, a, b)] = (round(float(np.mean(ch)), 3), round(max(ch), 3))
        print("%-26s seeds %s vs %s: mean %.3f  max-chunk %.3f" % (task, a, b, np.mean(ch), max(ch)), flush=True)
json.dump(res, open(S + "/lingbot-world-v2/specwm_out/spec_v0/base_selfdist.json", "w"), indent=1)
