"""Does the latent critic (trained only on draft-vs-target pairs) predict HUMAN-judged drift? Score the chunks of existing 1.3B-alone (never1p3b) and
14B-alone (never) clips: VAE-encode the video -> 3-latent chunks -> critic(x, prev, cam) per chunk -> clip score = min / mean logit; AUC vs blind severe labels
(paper/overnight/blind_fast1p3b). Also reports ③ is not available here (needs the models), so only the critic and simple latent stats.
usage: CUDA_VISIBLE_DEVICES=g python scripts/critic_eval_blind.py CRITIC.pt OUT.json"""
import os, sys, json, glob, collections, numpy as np, torch, imageio
from sklearn.metrics import roc_auc_score
S = "/home/hk28965/specwm"; V0 = S + "/lingbot-world-v2/specwm_out/spec_v0"; B = S + "/paper/overnight/blind_fast1p3b"
critic_path, out_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding; from specwm.critic_model import CriticScorer
SEV = {"teleport", "color_collapse", "texture_collapse", "freeze"}
mp = json.load(open(B + "/mapping.json")); R = {r: {x["id"]: x for f in sorted(glob.glob("%s/r%d_p*.json" % (B, r))) for x in json.load(open(f))} for r in (1, 2, 3)}
sev = lambda x: x["drift"] and bool(set(x.get("categories", [])) & SEV)
labels = {(v["task"], v["seed"], v["arm"]): sum(sev(R[r][i]) for r in (1, 2, 3)) >= 2 for i, v in mp.items()}
dev = torch.device("cuda:0"); FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
D = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), cfg_name="i2v-1.3B", assets_dir=FAST)
sc = CriticScorer(critic_path, device="cuda:0"); TT = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
res = {}; cams = {}
for (task, seed, arm), severe in sorted(labels.items()):
    if task not in cams:
        T = TT[task]; D.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); cams[task] = [e.squeeze(0).clone() for e in D._emb_chunks]
    f = "%s/%s/%s_s%d/video.mp4" % (V0, task, arm, seed)
    fr = np.stack([np.asarray(x) for x in imageio.get_reader(f)])[:153]
    v = torch.from_numpy(fr).float().div(127.5).sub(1).permute(3, 0, 1, 2)
    h, w = D._chunk_shape[2] * 8, D._chunk_shape[3] * 8
    if v.shape[2:] != (h, w): v = torch.nn.functional.interpolate(v.permute(1, 0, 2, 3), size=(h, w), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
    with torch.no_grad(): lat = D.pipe.vae.encode([v.to(dev)])[0].float()   # [16, 39, h/8, w/8]
    nC = lat.shape[1] // 3; logits = []; prev = None
    for ci in range(nC):
        x = lat[:, ci * 3:(ci + 1) * 3]; l, _ = sc(x, prev, cams[task][min(ci, len(cams[task]) - 1)]); logits.append(l); prev = x
    res["%s|%s|%d" % (task, arm, seed)] = dict(severe=severe, logits=[round(l, 3) for l in logits], min=min(logits), mean=float(np.mean(logits)), min_after0=min(logits[1:]))
    print("%-24s %-10s s%-4d severe=%d  min %.2f mean %.2f  %s" % (task[:24], arm, seed, severe, min(logits), np.mean(logits), " ".join("%+.1f" % l for l in logits)), flush=True)
json.dump(res, open(out_path, "w"), indent=1)
for arm in ("never1p3b", "never", None):
    rows = [v for k, v in res.items() if arm is None or k.split("|")[1] == arm]; y = [r["severe"] for r in rows]
    if len(set(y)) < 2: print("arm", arm, "n", len(rows), "labels all", y[0] if y else None); continue
    print("arm %-10s n %d severe %d | AUC(-min) %.3f AUC(-mean) %.3f AUC(-min_after0) %.3f" % (arm, len(rows), sum(y), roc_auc_score(y, [-r["min"] for r in rows]), roc_auc_score(y, [-r["mean"] for r in rows]), roc_auc_score(y, [-r["min_after0"] for r in rows])))
# the critic's draft-vs-target separation on these clips (1.3B-alone chunks should score lower than 14B-alone chunks)
a13 = [r["mean"] for k, r in res.items() if k.split("|")[1] == "never1p3b"]; a14 = [r["mean"] for k, r in res.items() if k.split("|")[1] == "never"]
print("mean logit: 1.3B-alone clips %.2f  14B-alone clips %.2f  AUC(14B>1.3B) %.3f" % (np.mean(a13), np.mean(a14), roc_auc_score([0] * len(a13) + [1] * len(a14), a13 + a14)))
