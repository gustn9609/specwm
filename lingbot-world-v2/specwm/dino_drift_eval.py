"""Zero-shot drift detector from DINOv2: decode each human-labelled clip (exp/drift_lat, severe = teleport / colour or texture collapse / freeze),
embed frames with DINOv2-L (CLS, normalized), and score drift as the drop of cosine similarity to the FIRST frame (the given scene).
Per clip: s_min = min over chunks of cos(chunk mean, frame 0); s_last; s_drop = 1 - s_min; also a mean over patch tokens variant and a
'collapse' feature (cos between consecutive chunks). AUROC vs the human 'severe' label by scene group, plus the cost per chunk.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.dino_drift_eval --out DIR [--n-other 320 --n-jc 200]"""
import argparse, os, sys, glob, json, time, random, numpy as np, torch, torch.nn.functional as F
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--n-other", type=int, default=320); ap.add_argument("--n-jc", type=int, default=200); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.makedirs(a.out, exist_ok=True)
from wan.modules.vae2_1 import Wan2_1_VAE
from transformers import AutoImageProcessor
from transformers.models.dinov2.modeling_dinov2 import Dinov2Model
from sklearn.metrics import roc_auc_score
dev = "cuda"; vae = Wan2_1_VAE(vae_pth=S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast/Wan2.1_VAE.pth", device=dev)
proc = AutoImageProcessor.from_pretrained(S + "/hf/dinov2-large"); dino = Dinov2Model.from_pretrained(S + "/hf/dinov2-large").to(dev).eval().half()
F_ = [f for f in sorted(glob.glob(S + "/exp/drift_lat/*.pt")) if "/cam_" not in f]; random.seed(0)
meta = [(f, torch.load(f, map_location="cpu", mmap=True)) for f in F_]; JC = ("sand_filling_up_jar", "sandcastle_wave_impact")
other = [m for m in meta if m[1]["task"] not in JC]; jc = [m for m in meta if m[1]["task"] in JC]; random.shuffle(other); random.shuffle(jc); sel = other[:a.n_other] + jc[:a.n_jc]
print("[dino] %d clips" % len(sel), flush=True)
@torch.no_grad()
def embed(frames):   # uint8 [T,H,W,3] -> CLS [T,D], patch-mean [T,D] (both L2-normalized)
    cls, pm = [], []
    for i in range(0, len(frames), 32):
        x = proc(images=list(frames[i:i + 32]), return_tensors="pt")["pixel_values"].to(dev).half(); h = dino(pixel_values=x).last_hidden_state.float()
        cls.append(F.normalize(h[:, 0], dim=-1).cpu()); pm.append(F.normalize(h[:, 1:].mean(1), dim=-1).cpu())
    return torch.cat(cls), torch.cat(pm)
recs = []; t_dec = t_emb = 0.0
for i, (f, d) in enumerate(sel):
    t0 = time.time()
    with torch.no_grad(): v = vae.decode([d["lat"].float().to(dev)])[0].clamp(-1, 1)
    fr = ((v + 1) * 127.5).permute(1, 2, 3, 0).byte().cpu().numpy(); torch.cuda.synchronize(); t_dec += time.time() - t0; t1 = time.time()
    cls, pm = embed(fr); torch.cuda.synchronize(); t_emb += time.time() - t1
    T = len(fr); nC = (T - 1) // 12 + 1; r = dict(task=d["task"], arm=d["arm"], seed=int(d["seed"]), severe=bool(d["severe"]), drift=bool(d["drift"]), cats=list(d.get("cats", [])), group="jc" if d["task"] in JC else "other", frames=T)
    for name, E in (("cls", cls), ("patch", pm)):
        c0 = E[0]; ch = [E[max(1, 12 * c + 1):12 * c + 13].mean(0) for c in range(nC)]; ch = [F.normalize(x, dim=0) for x in ch]
        s = [float(x @ c0) for x in ch]; cons = [float(ch[c] @ ch[c - 1]) for c in range(1, nC)]
        r[name] = dict(s_first=s, s_min=min(s), s_last=s[-1], s_mean=float(np.mean(s)), cons_min=min(cons) if cons else 1.0, drop_max=float(max(0.0, max(s[c - 1] - s[c] for c in range(1, nC))) if nC > 1 else 0.0))
    recs.append(r)
    if i % 50 == 0: print("[dino] %d/%d %s severe=%s cls s_min %.3f" % (i, len(sel), d["task"], d["severe"], r["cls"]["s_min"]), flush=True); json.dump(recs, open(a.out + "/records.json", "w"))
json.dump(recs, open(a.out + "/records.json", "w"))
nchunks = sum((r["frames"] - 1) // 12 + 1 for r in recs)
print("| scene group | n | severe rate | score | AUROC vs severe ↑ | AUROC vs any-drift ↑ |"); print("|---|---|---|---|---|---|")
for g, lab in (("other", "32 scenes never seen by the latent critic"), ("jc", "jar / castle"), ("all", "all")):
    R = [r for r in recs if g == "all" or r["group"] == g]; y = [r["severe"] for r in R]; yd = [r["drift"] for r in R]
    au = lambda yy, sc: roc_auc_score(yy, sc) if 0 < sum(yy) < len(yy) else float("nan")
    for feat in ("cls", "patch"):
        for key, sign in (("s_min", -1), ("s_last", -1), ("s_mean", -1), ("cons_min", -1), ("drop_max", 1)):
            sc = [sign * r[feat][key] for r in R]; print("| %s | %d | %.0f %% | %s %s | %.2f | %.2f |" % (lab, len(R), 100 * np.mean(y), feat, key, au(y, sc), au(yd, sc)))
print("cost: VAE decode %.1f ms/chunk, DINOv2-L embed %.1f ms/chunk (12 frames; would use 2-3 frames online)" % (1000 * t_dec / nchunks, 1000 * t_emb / nchunks))
print("[DINO_DONE]", flush=True)
