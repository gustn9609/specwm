"""Can the full-res latent drift critic (drift/noprev_noplanrc, human severe-drift labels, canonical 56x104 latent grid) judge low-res clips?
Three input routes per clip: (a) generated latents as they are (full-res lat.pt, or low-res lat.pt bilinearly canonicalized by the critic),
(b) the decoded (and, for low-res, Real-ESRGAN-upscaled 832x480) video re-encoded by the VAE. Sanity check = (a) vs (b) on full-res clips
(VAE round trip only). Reports per arm: mean logit, % chunks flagged (logit > tau), % clips with >= 1 flagged chunk, and per-clip logits.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.critic_lowres_eval --tasks a,b --seeds 4321,9999 --arms "label=ROOT/%s/ARM_s%d[,sr=ROOT_sr/%s/ARM_s%d]" ... --out DIR"""
import argparse, os, sys, json, glob, numpy as np, torch, imageio
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--tasks", required=True); ap.add_argument("--seeds", default="4321,9999"); ap.add_argument("--arms", nargs="+", required=True); ap.add_argument("--tau", type=float, default=0.05)
ap.add_argument("--critic", default=S + "/lingbot-world-v2/specwm_out/drift/noprev_noplanrc/critic.pt"); ap.add_argument("--out", required=True); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm.critic_model import CriticScorer
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
CS = CriticScorer(a.critic, device="cuda:0"); dev = "cuda:0"; os.makedirs(a.out, exist_ok=True)
@torch.no_grad()
def encode_video(path):
    fr = np.stack([np.asarray(f) for f in imageio.get_reader(path)]); v = torch.from_numpy(fr).permute(3, 0, 1, 2).float().div(127.5).sub(1)   # [3,T,H,W]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16): lat = B.pipe.vae.encode([v.to(dev)])[0].float()
    return lat.cpu()   # [16, F, h, w]
@torch.no_grad()
def score(lat, cam_chunks):
    F_ = lat.shape[1] // 3; out = []
    for c in range(F_):
        x = lat[:, 3 * c:3 * c + 3]; prev = lat[:, 3 * c - 3:3 * c] if c > 0 else None; l, _ = CS(x, prev, cam_chunks[min(c, len(cam_chunks) - 1)].squeeze(0)); out.append(l)
    return out
arms = []
for spec in a.arms:
    d = dict(kv.split("=", 1) for kv in spec.split(",")); arms.append((d.pop("label"), d))
recs = []
for task in a.tasks.split(","):
    T = ALL[task]
    with torch.no_grad(): B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832)
    cam = [e.detach() for e in B._emb_chunks]
    for seed in [int(s) for s in a.seeds.split(",")]:
        for label, d in arms:
            r = dict(task=task, seed=seed, arm=label)
            for route, pat in d.items():
                p = pat % (task, seed)
                if route == "lat" and os.path.exists(p + "/lat.pt"): r["lat"] = score(torch.load(p + "/lat.pt").float(), cam)
                elif route in ("video", "sr"):
                    vp = p + "/video.mp4"
                    if os.path.exists(vp): r[route] = score(encode_video(vp), cam)
            recs.append(r); print("[CRIT] %s s%d %s " % (task, seed, label) + " | ".join("%s: mean %.2f max %.2f flagged %d/%d" % (k, np.mean(v), np.max(v), sum(x > a.tau for x in v), len(v)) for k, v in r.items() if isinstance(v, list)), flush=True)
json.dump(recs, open(a.out + "/records.json", "w"))
print("| arm | route | mean logit ↓ | chunks flagged (> %.2f) ↓ | clips with ≥1 flagged | n clips |" % a.tau); print("|---|---|---|---|---|---|")
for label, d in arms:
    for route in ("lat", "video", "sr"):
        L = [r[route] for r in recs if r["arm"] == label and route in r]
        if not L: continue
        fl = [sum(x > a.tau for x in v) / len(v) for v in L]; print("| %s | %s | %.2f | %.0f %% | %.0f %% | %d |" % (label, {"lat": "generated latents", "video": "decoded video → VAE", "sr": "SR 832×480 video → VAE"}[route], np.mean([np.mean(v) for v in L]), 100 * np.mean(fl), 100 * np.mean([f > 0 for f in fl]), len(L)))
# sanity: lat vs video on the same clips
pairs = [(np.array(r["lat"]), np.array(r["video"])) for r in recs if "lat" in r and "video" in r]
if pairs: print("sanity (full-res clips, same chunk): mean |logit(lat) - logit(video)| = %.3f over %d chunks; Spearman %.2f" % (np.mean([np.abs(x - y).mean() for x, y in pairs]), sum(len(x) for x, _ in pairs), __import__("scipy.stats").stats.spearmanr(np.concatenate([x for x, _ in pairs]), np.concatenate([y for _, y in pairs]))[0]))
print("[CRIT_DONE]", flush=True)
