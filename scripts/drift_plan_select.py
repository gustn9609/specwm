"""Plan B preview with the drift critic: score the 88 v2-bid plan candidates (planrc, 22 cases x 4) and pick the lowest-drift plan per case;
compare with the blind labels of their renders (paper/overnight/blind_planrc/planrc_labels.json): clean renders /22 for the pick, vs k=0 (current), oracle, HSV, ③.
Plans are 576x304 clips (161 frames): VAE-encode, chunk into 3 latents, canon-resize inside the scorer.
usage: CUDA_VISIBLE_DEVICES=g python scripts/drift_plan_select.py CRITIC.pt OUT.json"""
import os, sys, json, glob, collections, numpy as np, torch, imageio
from sklearn.metrics import roc_auc_score
S = "/home/hk28965/specwm"; V0 = S + "/lingbot-world-v2/specwm_out/spec_v0"; critic_path, out_path = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding; from specwm.critic_model import CriticScorer
lab = json.load(open(S + "/paper/overnight/blind_planrc/planrc_labels.json")); sc3 = {"%s|%d|%s" % (x["task"], x["seed"], x["k"]): x for x in json.load(open(S + "/exp/kveval/planrc_scores.json"))}
feat = json.load(open(S + "/exp/kveval/planrc_features.json")) if os.path.exists(S + "/exp/kveval/planrc_features.json") else {}
dev = torch.device("cuda:0"); FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
D = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), cfg_name="i2v-1.3B", assets_dir=FAST)
sc = CriticScorer(critic_path, device="cuda:0"); ck = torch.load(critic_path, map_location="cpu"); temp = ck.get("temp", 2.0)
TT = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; cams = {}
res = {}
for key, l in sorted(lab.items()):
    t, s, k = key.split("|"); arm = "planv2_l4c" if k == "0" else "planv2c%s_l4c" % k; f = "%s/%s/%s_s%s/video.mp4" % (V0, t, arm, s)
    if t not in cams: T = TT[t]; D.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); cams[t] = [e.squeeze(0).clone() for e in D._emb_chunks]
    fr = np.stack([np.asarray(x) for x in imageio.get_reader(f)])[:157]
    v = torch.from_numpy(fr).float().div(127.5).sub(1).permute(3, 0, 1, 2)
    with torch.no_grad(): lat = D.pipe.vae.encode([v.to(dev)])[0].float()   # [16, 40, 38, 72] at plan res
    nC = min(lat.shape[1] // 3, 13); logits = []; prev = None
    for ci in range(nC):
        x = lat[:, ci * 3:(ci + 1) * 3]; lg, _ = sc(x, prev, cams[t][ci]); logits.append(lg); prev = x
    L = torch.tensor(logits); lse = float(temp * torch.logsumexp(L / temp, 0) - temp * np.log(len(L)))
    res[key] = dict(severe=not l["clean"], max=max(logits), lse=lse, logits=[round(x, 2) for x in logits])
    print("%-6s s%-4s k%s severe=%d  max %+.2f lse %+.2f" % (t[:6], s, k, not l["clean"], max(logits), lse), flush=True)
json.dump(res, open(out_path, "w"), indent=1)
y = [r["severe"] for r in res.values()]
print("AUC(plan drift-critic -> render severe): max %.3f  lse %.3f" % (roc_auc_score(y, [r["max"] for r in res.values()]), roc_auc_score(y, [r["lse"] for r in res.values()])))
cases = sorted({(k.split("|")[0], k.split("|")[1]) for k in res}); cnt = collections.Counter()
for t, s in cases:
    ks = [k for k in res if k.startswith("%s|%s|" % (t, s))]
    for name, f in (("critic max", lambda k: res[k]["max"]), ("critic lse", lambda k: res[k]["lse"]), ("③ m0.3", lambda k: sc3[k]["m0.3"]), ("HSV value", lambda k: feat[k]["val"] if feat else 0)):
        cnt[name] += not res[min(ks, key=f)]["severe"]
    cnt["k=0 current"] += not res["%s|%s|0" % (t, s)]["severe"]; cnt["oracle"] += any(not res[k]["severe"] for k in ks)
print("clean renders / %d cases when picking the plan by: " % len(cases) + ", ".join("%s %d" % (k, v) for k, v in cnt.items()))
