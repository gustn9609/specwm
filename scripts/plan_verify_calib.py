"""Offline calibration of a render-vs-plan verifier: per 12-frame chunk, DINO distance between the render (downsampled to plan res) and the plan
chunk + HSV colour shift; clip score = max over chunks. AUC vs blind 'severe' labels. usage: plan_verify_calib.py OUT.json"""
import sys, json, numpy as np, torch, imageio, cv2
from sklearn.metrics import roc_auc_score
S = "/home/hk28965/specwm"; V = S + "/lingbot-world-v2/specwm_out/spec_v0"; dev = torch.device("cuda:0")
torch.hub.set_dir(S + "/.torch_hub"); dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval().to(dev)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
@torch.no_grad()
def feats(V_):
    x = torch.from_numpy(np.stack(V_)).to(dev).permute(0, 3, 1, 2).float() / 255; x = torch.nn.functional.interpolate(x, size=(224, 392), mode="bilinear", antialias=True)
    return torch.nn.functional.normalize(dino((x - MEAN) / STD).float(), dim=-1).cpu()
def sv(fr): h = [cv2.cvtColor(f, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32) for f in fr]; return np.array([np.mean([x[:, 1].mean() for x in h]), np.mean([x[:, 2].mean() for x in h])])
L = json.load(open(S + "/exp/kveval/blind_lookup.json")); SEV = {"teleport", "color_collapse", "texture_collapse", "freeze"}
out = {}
for t in ("sand_filling_up_jar", "sandcastle_wave_impact"):
    for s in (1234, 4321, 9999, 11, 22, 33, 44, 55):
        P = [np.asarray(f) for f in imageio.get_reader("%s/%s/plan_l4c_s%d/video.mp4" % (V, t, s))]; Pc = [np.asarray(f) for f in imageio.get_reader("%s/%s/plan_cr_s%d/video.mp4" % (V, t, s))]
        for arm, plan in (("plh_plan6_ck400_l4c_st0_never", P), ("plh_plan6_ck400_cr_st0_never", Pc), ("never", P)):
            R = [np.asarray(f) for f in imageio.get_reader("%s/%s/%s_s%d/video.mp4" % (V, t, arm, s))]; h, w = plan[0].shape[:2]
            Rs = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in R]; n = min(len(Rs), len(plan))
            fr, fp = feats(Rs[:n]), feats(plan[:n]); dd, dc = [], []
            for c in range(1, (n - 1) // 12 + 1):
                ix = list(range(12 * (c - 1) + 1, min(12 * c + 1, n)))
                a = torch.nn.functional.normalize(fr[ix].mean(0), dim=0); b = torch.nn.functional.normalize(fp[ix].mean(0), dim=0)
                dd.append(float(1 - a @ b)); dc.append(float(np.abs(sv([Rs[i] for i in ix]) - sv([plan[i] for i in ix])).sum() / 80))
            lab = L.get("%s|%s|%d" % (t, arm, s)); sev = bool(lab and lab["drift"] and any(v >= 2 and c in SEV for c, v in lab["cats"].items()))
            out["%s|%s|%d" % (t, arm, s)] = dict(d=dd, c=dc, severe=sev)
            print("%s %-30s s%-5d severe=%d  maxD %.3f  maxC %.3f  D/chunk %s" % (t[:6], arm[:30], s, sev, max(dd), max(dc), " ".join("%.2f" % x for x in dd)), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
for name, keep in (("v2 renders only", lambda k: "plh_" in k), ("v2 + Fast alone", lambda k: True)):
    ks = [k for k in out if keep(k)]; y = [out[k]["severe"] for k in ks]
    for sc, f in (("max DINO", lambda o: max(o["d"])), ("max DINO + colour", lambda o: max(np.array(o["d"]) + np.array(o["c"]))), ("mean DINO", lambda o: np.mean(o["d"]))):
        print("%-16s %-18s n=%d severe=%d AUC %.3f" % (name, sc, len(ks), sum(y), roc_auc_score(y, [f(out[k]) for k in ks]) if 0 < sum(y) < len(y) else float("nan")))
