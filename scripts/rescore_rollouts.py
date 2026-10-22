"""Re-score saved draft rollouts with another drift critic (latent only; cam_err kept): rewrites drift/reward in place (old values kept as drift_old/reward_old).
usage: CUDA_VISIBLE_DEVICES=g python scripts/rescore_rollouts.py CRITIC.pt DIR [DIR ...]"""
import sys, os, glob, math, json, torch
S = "/home/hk28965/specwm"; sys.path.insert(0, S + "/lingbot-world-v2")
from specwm.critic_model import CriticScorer, pool_cam
crit_path = os.path.abspath(sys.argv[1]); sc = CriticScorer(crit_path, device="cuda:0"); temp = torch.load(crit_path, map_location="cpu").get("temp", 2.0)
cams = {}
B = {}
def cam_of(task):
    if task not in cams:
        f = S + "/exp/drift_lat/cam_%s.pt" % task
        if os.path.exists(f): cams[task] = torch.load(f, map_location="cpu")["cam"]
        else:   # compute the Plucker embeddings with the 1.3B binding (same prepare as the rollouts) and cache them
            if "b" not in B:
                import json; os.chdir(S + "/lingbot-world-v2"); from specwm.binding_v2 import V2Binding
                B["b"] = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), cfg_name="i2v-1.3B", assets_dir=S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast")
                B["T"] = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
            T = B["T"][task]; B["b"].prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832)
            cams[task] = torch.stack([e.squeeze(0).clone() for e in B["b"]._emb_chunks]).half().cpu(); torch.save(dict(cam=cams[task], shape=list(B["b"]._chunk_shape)), f)
    return cams[task]
n = 0; miss = set()
for d in sys.argv[2:]:
    for f in glob.glob(d + "/*.pt"):
        r = torch.load(f, map_location="cpu"); c = cam_of(r["task"])
        if c is None: miss.add(r["task"]); continue
        lat = r["lat"].float(); L = torch.tensor([sc(lat[:, 3 * k:3 * k + 3], None, c[k])[0] for k in range(lat.shape[1] // 3)])
        drift = float(temp * torch.logsumexp(L / temp, 0) - temp * math.log(len(L)))
        r.setdefault("drift_old", r["drift"]); r.setdefault("reward_old", r["reward"]); r["drift"] = drift; r["reward"] = -drift / 4.0 - r["cam_err"] / 30.0; r["critic"] = crit_path
        torch.save(r, f); n += 1
print("[RESCORE_DONE] %d rollouts rescored with %s; tasks without cached camera emb: %s" % (n, crit_path, sorted(miss)))
