"""Cheap plan-level features for the 88 plan candidates vs their renders' blind labels (planrc_labels.json).
Per plan (12-frame chunks): start = max_c DINO dist(chunk, start frame); loop = DINO dist of chunk 11 to start; nov = max_c min dist to earlier chunks + start;
sat = min_c mean HSV-S(chunk)/S(start); val = max_c |V(chunk)/V(start) - 1|; ir_min = min frame ImageReward; ir_c = min over chunks of mean ImageReward.
Prints AUC (feature -> render severe) and the selection table (pick the candidate with the best feature per case). usage: CUDA_VISIBLE_DEVICES=g planrc_features.py OUT.json"""
import sys, json, collections, numpy as np, torch, imageio, cv2
from PIL import Image
from sklearn.metrics import roc_auc_score
S = "/home/hk28965/specwm"; V = S + "/lingbot-world-v2/specwm_out/spec_v0"; B = S + "/paper/overnight/blind_planrc"; dev = torch.device("cuda:0")
torch.hub.set_dir(S + "/.torch_hub"); dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval().to(dev)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
import ImageReward as RM; irm = RM.load("ImageReward-v1.0", device="cuda"); irm.eval()
PR = {t: open("%s/exp/irprompt_%s.txt" % (S, t)).read().strip() for t in ("sand_filling_up_jar", "sandcastle_wave_impact")}
@torch.no_grad()
def emb(fr):
    x = torch.from_numpy(np.stack(fr)).to(dev).permute(0, 3, 1, 2).float() / 255; x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return torch.nn.functional.normalize(dino((x - MEAN) / STD).float(), dim=-1).cpu()
def sv(fr): h = [cv2.cvtColor(f, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32) for f in fr]; return np.mean([x[:, 1].mean() for x in h]), np.mean([x[:, 2].mean() for x in h])
lab = json.load(open(B + "/planrc_labels.json")); out = {}
for key, l in lab.items():
    t, s, k = key.split("|"); arm = "planv2_l4c" if k == "0" else "planv2c%s_l4c" % k
    fr = [np.asarray(f) for f in imageio.get_reader("%s/%s/%s_s%s/video.mp4" % (V, t, arm, s))][:157]
    E = emb(fr); e0 = E[0]; ch = [list(range(1 + 12 * c, 1 + 12 * (c + 1))) for c in range(13)]
    d = lambda a, b: float(1 - (a @ b))
    start = max(np.mean([d(E[i], e0) for i in c]) for c in ch); loop = np.mean([d(E[i], e0) for i in ch[11]])
    nov = 0.0
    for ci, c in enumerate(ch):
        prev = [0] + [i for cc in ch[:ci] for i in cc]; nov = max(nov, np.mean([min(d(E[i], E[j]) for j in prev) for i in c]))
    s0, v0 = sv(fr[:1]); sats, vals = zip(*[sv([fr[i] for i in c[::3]]) for c in ch])
    with torch.no_grad(): ir = [irm.score(PR[t], Image.fromarray(fr[i])) for i in range(0, len(fr), 2)]
    irc = [np.mean(ir[c[0] // 2:(c[-1] // 2) + 1]) for c in ch]
    out[key] = dict(severe=not l["clean"], start=float(start), loop=float(loop), nov=float(nov), sat=float(min(sats) / s0), val=float(max(abs(v / v0 - 1) for v in vals)), ir_min=float(min(ir)), ir_c=float(min(irc)))
    print("%-40s sev=%d start %.2f loop %.2f nov %.2f sat %.2f val %.2f ir %.2f/%.2f" % (key, out[key]["severe"], start, loop, nov, out[key]["sat"], out[key]["val"], out[key]["ir_min"], out[key]["ir_c"]), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
y = [v["severe"] for v in out.values()]
# sign: higher start/loop/nov/val -> more drift; lower sat/ir -> more drift
for f, sg in (("start", 1), ("loop", 1), ("nov", 1), ("sat", -1), ("val", 1), ("ir_min", -1), ("ir_c", -1)): print("AUC %-7s %.3f" % (f, roc_auc_score(y, [sg * v[f] for v in out.values()])))
cases = sorted({(k.split("|")[0], k.split("|")[1]) for k in out}); res = collections.Counter()
for f, sg in (("start", 1), ("loop", 1), ("nov", 1), ("sat", -1), ("val", 1), ("ir_min", -1), ("ir_c", -1)):
    for t, s in cases:
        ks = [k for k in out if k.startswith("%s|%s|" % (t, s))]; best = min(ks, key=lambda k: sg * out[k][f]); res[f] += not out[best]["severe"]
print("pick-by-feature clean renders / %d cases: " % len(cases) + ", ".join("%s %d" % (f, res[f]) for f in ("start", "loop", "nov", "sat", "val", "ir_min", "ir_c")))
