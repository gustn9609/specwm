"""Label-free drift score for finished videos: DINOv2-L CLS cosine between each chunk (12 frames) and frame 0; reports per arm the mean of the
per-clip minimum (s_min, higher = scene kept) and the mean cosine, plus the share of chunks below 0.5. usage: python scripts/dino_score_videos.py LABEL=GLOB ..."""
import sys, glob, os, numpy as np, torch, torch.nn.functional as F, cv2
from transformers import AutoImageProcessor
from transformers.models.dinov2.modeling_dinov2 import Dinov2Model
S = "/home/hk28965/specwm"; dev = "cuda"
proc = AutoImageProcessor.from_pretrained(S + "/hf/dinov2-large"); dino = Dinov2Model.from_pretrained(S + "/hf/dinov2-large").to(dev).eval().half()
@torch.no_grad()
def embed(frames):
    out = []
    for i in range(0, len(frames), 32):
        x = proc(images=frames[i:i + 32], return_tensors="pt")["pixel_values"].to(dev).half(); out.append(F.normalize(dino(pixel_values=x).last_hidden_state[:, 0].float(), dim=-1).cpu())
    return torch.cat(out)
def frames(p):
    cap = cv2.VideoCapture(p); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return fr
print("| arm | clips | DINO cos to frame 0: min over chunks ↑ | mean ↑ | chunks with cos < 0.5 ↓ |"); print("|---|---|---|---|---|")
for spec in sys.argv[1:]:
    lab, pat = spec.split("=", 1); smin, smean, low, n = [], [], [], 0
    for p in sorted(glob.glob(pat)):
        fr = frames(p)
        if len(fr) < 24: continue
        E = embed(fr); c0 = E[0]; nC = (len(fr) - 1) // 12 + 1; s = [float(F.normalize(E[max(1, 12 * c + 1):12 * c + 13].mean(0), dim=0) @ c0) for c in range(nC)]
        smin.append(min(s)); smean.append(np.mean(s)); low += [x < 0.5 for x in s]; n += 1
    if n: print("| %s | %d | %.3f | %.3f | %.0f %% |" % (lab, n, np.mean(smin), np.mean(smean), 100 * np.mean(low)))
