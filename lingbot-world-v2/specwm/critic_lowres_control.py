"""Calibration control for critic_lowres_eval: apply the low-res pipeline's degradation to FULL-RES clips (downscale to 576x320 / 448x256, then
either VAE-encode at that size (latent route) or Real-ESRGAN back to 832x480 and VAE-encode (SR route)) and score with the drift critic.
Whatever the logit rises by here is domain shift, not drift. usage: CUDA_VISIBLE_DEVICES=g python -m specwm.critic_lowres_control --tasks a,b --seeds 1234,4321,9999 --out DIR"""
import argparse, os, sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, imageio
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--tasks", required=True); ap.add_argument("--seeds", default="1234,4321,9999"); ap.add_argument("--tau", type=float, default=0.05); ap.add_argument("--out", required=True); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm.critic_model import CriticScorer
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"; O = S + "/lingbot-world-v2/specwm_out"
B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
CS = CriticScorer(O + "/drift/noprev_noplanrc/critic.pt", device="cuda:0"); dev = "cuda:0"; os.makedirs(a.out, exist_ok=True)
class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
        super().__init__(); self.upscale = upscale; self.body = nn.ModuleList([nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)])
        for _ in range(num_conv): self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1)); self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1)); self.upsampler = nn.PixelShuffle(upscale)
    def forward(self, x):
        out = x
        for m in self.body: out = m(out)
        return self.upsampler(out) + F.interpolate(x, scale_factor=self.upscale, mode="nearest")
SR = SRVGGNetCompact().to(dev).eval().half(); SR.load_state_dict(torch.load(S + "/hf/realesrgan/realesr-general-x4v3.pth", map_location="cpu")["params"], strict=True)
@torch.no_grad()
def encode(v):   # [3,T,H,W] in [-1,1]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16): return B.pipe.vae.encode([v.to(dev)])[0].float().cpu()
@torch.no_grad()
def sr_up(v01, TH, TW):   # [T,3,h,w] in [0,1] -> [T,3,TH,TW]
    out = []
    for i in range(0, v01.shape[0], 8):
        y = SR(v01[i:i + 8].to(dev).half()).clamp(0, 1); out.append(F.interpolate(y, size=(TH, TW), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1).float().cpu())
    return torch.cat(out)
@torch.no_grad()
def score(lat, cam):
    return [CS(lat[:, 3 * c:3 * c + 3], lat[:, 3 * c - 3:3 * c] if c > 0 else None, cam[min(c, len(cam) - 1)].squeeze(0))[0] for c in range(lat.shape[1] // 3)]
RES = {"576": (320, 576), "448": (256, 448)}; ARMS = {"14B 4-step 832": O + "/early_cmp/%s/uniform4_s%d/video.mp4", "14B 2-step 832": O + "/early_cmp/%s/uniform2_s%d/video.mp4", "1.3B draft 832": O + "/pipe_ff3/%s/draft_1p3b_s%d/video.mp4"}
recs = []
for task in a.tasks.split(","):
    T = ALL[task]
    with torch.no_grad(): B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832)
    cam = [e.detach() for e in B._emb_chunks]
    for seed in [int(s) for s in a.seeds.split(",")]:
        for label, pat in ARMS.items():
            p = pat % (task, seed)
            if not os.path.exists(p): continue
            fr = torch.from_numpy(np.stack([np.asarray(f) for f in imageio.get_reader(p)])).permute(0, 3, 1, 2).float() / 255   # [T,3,H,W]
            TH, TW = fr.shape[2:]; r = dict(task=task, seed=seed, arm=label, orig=score(encode(fr.permute(1, 0, 2, 3) * 2 - 1), cam))
            for k, (h, w) in RES.items():
                dn = F.interpolate(fr, size=(h, w), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
                r["lat" + k] = score(encode(dn.permute(1, 0, 2, 3) * 2 - 1), cam); r["sr" + k] = score(encode(sr_up(dn, TH, TW).permute(1, 0, 2, 3) * 2 - 1), cam)
            recs.append(r); print("[CTRL] %s s%d %s " % (task, seed, label) + " ".join("%s %.2f" % (k, np.mean(v)) for k, v in r.items() if isinstance(v, list)), flush=True)
json.dump(recs, open(a.out + "/records.json", "w"))
print("| full-res clip, degraded as the low-res pipeline would | mean logit | chunks flagged (> %.2f) | Δ logit vs original |" % a.tau); print("|---|---|---|---|")
for k, lab in (("orig", "original 832×480 (VAE round trip)"), ("lat576", "↓576×320 → VAE at 576 (latent route)"), ("sr576", "↓576×320 → SR → VAE at 832 (SR route)"), ("lat448", "↓448×256 → VAE at 448 (latent route)"), ("sr448", "↓448×256 → SR → VAE at 832 (SR route)")):
    L = [r[k] for r in recs]; print("| %s | %.2f | %.0f %% | %+.2f |" % (lab, np.mean([np.mean(v) for v in L]), 100 * np.mean([np.mean([x > a.tau for x in v]) for v in L]), np.mean([np.mean(v) for v in L]) - np.mean([np.mean(r["orig"]) for r in recs])))
print("[CTRL_DONE]", flush=True)
