"""Can a cheap check on the low-res PLAN predict the failure of its render? Scores per plan (chunk = 12 frames):
start = max_c DINO distance(chunk, start image); novelty = max_c min distance(chunk, all earlier chunks + start); colour = max_c HSV shift vs start;
loop = distance of chunk 11 (camera back at the start view, frames 121-132) to the start image. AUC vs the render's blind 'severe' label."""
import json, numpy as np, torch, imageio, cv2
from sklearn.metrics import roc_auc_score
S = "/home/hk28965/specwm"; V = S + "/lingbot-world-v2/specwm_out/spec_v0"; dev = torch.device("cuda:0")
torch.hub.set_dir(S + "/.torch_hub"); dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval().to(dev)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
@torch.no_grad()
def feats(V_):
    x = torch.from_numpy(np.stack(V_)).to(dev).permute(0, 3, 1, 2).float() / 255; x = torch.nn.functional.interpolate(x, size=(224, 392), mode="bilinear", antialias=True)
    return torch.nn.functional.normalize(dino((x - MEAN) / STD).float(), dim=-1).cpu()
def sv(fr): h = [cv2.cvtColor(f, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32) for f in fr]; return np.array([np.mean([x[:, 1].mean() for x in h]), np.mean([x[:, 2].mean() for x in h])])
C = json.load(open(S + "/exp/kveval/plan_verify_calib.json")); rows = []
for t in ("sand_filling_up_jar", "sandcastle_wave_impact"):
    for s in (1234, 4321, 9999, 11, 22, 33, 44, 55):
        for tag, arm in (("l4c", "plh_plan6_ck400_l4c_st0_never"), ("cr", "plh_plan6_ck400_cr_st0_never")):
            P = [np.asarray(f) for f in imageio.get_reader("%s/%s/plan_%s_s%d/video.mp4" % (V, t, tag, s))]; F = feats(P); n = len(P)
            ch = [list(range(12 * (c - 1) + 1, min(12 * c + 1, n))) for c in range(1, (n - 1) // 12 + 1)]
            cf = [torch.nn.functional.normalize(F[i].mean(0), dim=0) for i in ch]; f0 = F[0]
            start = [float(1 - c @ f0) for c in cf]; nov = [float(min([1 - c @ f0] + [1 - c @ cf[j] for j in range(k)])) for k, c in enumerate(cf)]
            col = [float(np.abs(sv([P[i] for i in ix]) - sv([P[0]])).sum() / 80) for ix in ch]
            rows.append(dict(key="%s|%s|%d" % (t[:6], tag, s), severe=C["%s|%s|%d" % (t, arm, s)]["severe"], start=max(start), nov=max(nov), col=max(col), loop=start[10], novcol=max(np.array(nov) + np.array(col))))
y = [r["severe"] for r in rows]
for k in ("start", "nov", "col", "novcol", "loop"): print("%-7s AUC %.3f" % (k, roc_auc_score(y, [r[k] for r in rows])))
for r in sorted(rows, key=lambda r: -r["novcol"]): print("%-18s severe=%d nov %.2f col %.2f loop %.2f" % (r["key"], r["severe"], r["nov"], r["col"], r["loop"]))
json.dump(rows, open(S + "/exp/kveval/plan_only_calib.json", "w"), indent=1)
