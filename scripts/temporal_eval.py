#!/usr/bin/env python3
"""Strict temporal-consistency evaluation of rollouts against Base 70-step.
Per video (chunk = 12 frames after chunk 0's 9): motion (mean |dF|), flicker (mean |F[i+1]-2F[i]+F[i-1]|), boundary jump ratio
(step into each chunk / median step inside chunks), SSIM between frames 4 apart (background stability), DINO CLS drift
between consecutive chunks (scene change), DINO distance of each chunk to the start frame and to Base 70-step's same chunk.
usage: temporal_eval.py OUT.json RUN_DIR [RUN_DIR ...]   (RUN_DIR = .../<task>/<run>_s<seed>; base70 is found next to it)"""
import sys, os, json, numpy as np, torch, imageio, cv2
S = "/home/hk28965/specwm"; dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.hub.set_dir(S + "/.torch_hub"); dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval().to(dev)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
@torch.no_grad()
def feats(V):
    out = []
    for i in range(0, len(V), 16):
        x = torch.from_numpy(np.stack(V[i:i + 16])).to(dev).permute(0, 3, 1, 2).float() / 255; x = torch.nn.functional.interpolate(x, size=(224, 392), mode="bilinear", antialias=True)
        out.append(torch.nn.functional.normalize(dino((x - MEAN) / STD).float(), dim=-1).cpu())
    return torch.cat(out)
def ssim(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64); b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2; g = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    ma, mb = g(a), g(b); va, vb, vab = g(a * a) - ma * ma, g(b * b) - mb * mb, g(a * b) - ma * mb
    return float((((2 * ma * mb + C1) * (2 * vab + C2)) / ((ma * ma + mb * mb + C1) * (va + vb + C2))).mean())
def chunks(n): return [list(range(0, 9))] + [list(range(12 * c - 3, min(12 * c + 9, n))) for c in range(1, 13)]
def analyse(path, base_f=None):
    V = [np.asarray(f) for f in imageio.get_reader(path)]; n = len(V); Vf = [v.astype(np.float32) for v in V]
    d1 = np.array([np.abs(Vf[i + 1] - Vf[i]).mean() for i in range(n - 1)]); d2 = np.array([np.abs(Vf[i + 1] - 2 * Vf[i] + Vf[i - 1]).mean() for i in range(1, n - 1)])
    F = feats(V); C = chunks(n); cf = torch.stack([torch.nn.functional.normalize(F[c].mean(0), dim=0) for c in C])
    bidx = [12 * c - 4 for c in range(1, 13) if 12 * c - 4 < n - 1]; inside = np.median([d1[i] for i in range(n - 1) if i not in bidx])
    r = dict(motion=float(d1.mean()), flicker=float(d2.mean()), boundary_ratio=[round(float(d1[i] / inside), 2) for i in bidx],
             ssim4=float(np.mean([ssim(V[i], V[i + 4]) for i in range(0, n - 4, 2)])),
             chunk_drift=[round(float(1 - cf[c] @ cf[c - 1]), 3) for c in range(1, len(C))],
             to_start=[round(float(1 - cf[c] @ F[0]), 3) for c in range(len(C))])
    if base_f is not None: r["to_base"] = [round(float(1 - cf[c] @ base_f[c]), 3) for c in range(min(len(C), len(base_f)))]
    return r, cf
out = {}
for run in sys.argv[2:]:
    task, name = run.rstrip("/").split("/")[-2:]; seed = name.rsplit("_s", 1)[1]; bp = os.path.join(os.path.dirname(run), "%s_s%s" % (os.environ.get("BASE_ARM", "base70"), seed), "video.mp4")
    rb, bf = analyse(bp) if os.path.exists(bp) else (None, None)
    r, _ = analyse(os.path.join(run, "video.mp4"), bf); r["gen_s"] = json.load(open(os.path.join(run, "spec_v0.json")))["gen_s"] if os.path.exists(os.path.join(run, "spec_v0.json")) else None
    out["%s/%s" % (task, name)] = r
    if rb is not None and "%s/%s_s%s" % (task, os.environ.get("BASE_ARM", "base70"), seed) not in out: out["%s/%s_s%s" % (task, os.environ.get("BASE_ARM", "base70"), seed)] = rb
    print("%-52s motion %5.1f flicker %5.1f ssim4 %.3f maxboundary %4.1fx  drift(max) %.2f  to_base(mean) %s" % ("%s/%s" % (task[:6], name), r["motion"], r["flicker"], r["ssim4"], max(r["boundary_ratio"]), max(r["chunk_drift"]), "%.2f" % np.mean(r["to_base"]) if "to_base" in r else "-"), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
