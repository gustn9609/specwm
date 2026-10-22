#!/usr/bin/env python3
"""Exp 3 / C2: anchor 1-frame correction. Fast rolls chunks 0-5 (frames 0..72); the single frame 72 is corrected (Base 1-frame SDEdit, done by hybrid_refine beforehand)
and Fast is RESTARTED from that frame as the new conditioning image with the remaining camera path (poses 72..156, 85 f = 7 chunks). Arms: anchor_fast (control: uncorrected
frame 72), anchor_base1f (Base-corrected frame 72), anchor_base70 (upper bound: Base70's own frame 72). Metric: DINO of generated frames 72..156 to Base70 frames 72..156 (= chunks 6-12).
usage: c2_anchor.py --scene 05 --seed 1234 --anchors anchor_base1f=<png>[,name=png] --out <dir>"""
import argparse, json, os, time, numpy as np, torch, imageio
from PIL import Image
S = "/home/hk28965/specwm"; R = S + "/lingbot-world-v2/specwm_out"
ap = argparse.ArgumentParser(); ap.add_argument("--scene", required=True); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--anchors", default="")
ap.add_argument("--anchor-frame", type=int, default=72); ap.add_argument("--frames", type=int, default=85); ap.add_argument("--out", required=True); ap.add_argument("--fast-video", default=None)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); sc, sd = a.scene, a.seed
fv = a.fast_video or R + "/e1repro/S%s/W18_sink6/fast_alone_s%d/video.mp4" % (sc, sd); bv = R + "/floor/base/S%s_s%d/video.mp4" % (sc, sd)
VF = np.stack([np.asarray(f) for f in imageio.get_reader(fv)]); VB = np.stack([np.asarray(f) for f in imageio.get_reader(bv)])
F0 = a.anchor_frame; anchors = {"anchor_fast": VF[F0], "anchor_base70": VB[F0]}
for kv in [x for x in a.anchors.split(",") if x]: n, p = kv.split("="); anchors[n] = np.asarray(Image.open(p).convert("RGB"))
# sliced action dir (poses re-based by compute_relative_poses inside the binding; intrinsics copied)
act = S + "/rt2_action_" + sc; ad = os.path.join(a.out, "action_from_%d" % F0); os.makedirs(ad, exist_ok=True)
P = np.load(act + "/poses.npy"); np.save(ad + "/poses.npy", P[F0:F0 + a.frames]); np.save(ad + "/intrinsics.npy", np.load(act + "/intrinsics.npy")[F0:F0 + a.frames])
for f in ("action.npy", "wasd_action.npy"):
    if os.path.exists(act + "/" + f): x = np.load(act + "/" + f); np.save(ad + "/" + f, x[F0:F0 + a.frames] if len(x) >= F0 + a.frames else x)
prompt = open(S + "/p_%s.txt" % sc).read().strip()
import sys; sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding
b = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
torch.hub.set_dir(S + "/.torch_hub"); dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval().cuda()
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda(); std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()
def feats(V):
    X = torch.from_numpy(np.ascontiguousarray(V)).permute(0, 3, 1, 2).float().cuda() / 255; X = torch.nn.functional.interpolate(X, size=(518, 518), mode="bilinear"); X = (X - mean) / std; o = []
    with torch.no_grad():
        for i in range(0, len(X), 8): o.append(torch.nn.functional.normalize(dino(X[i:i + 8]).flatten(1), dim=-1).cpu())
    return torch.cat(o)
def d(FA, FB): n = min(len(FA), len(FB)); return float((1 - (FA[:n] * FB[:n]).sum(-1)).mean())
FB_ref = feats(VB[F0:F0 + a.frames][::4]); FF_ref = feats(VF[F0:F0 + a.frames][::4])
rp = os.path.join(a.out, "c2_S%s_s%d.json" % (sc, sd)); rep = json.load(open(rp)) if os.path.exists(rp) else {}
rep["fast_alone_6_12_to_base"] = d(FF_ref, FB_ref)
for name, img in anchors.items():
    if name in rep and "to_base" in rep[name]: continue
    ip = os.path.join(a.out, "%s_S%s_s%d.png" % (name, sc, sd)); Image.fromarray(img).save(ip)
    cond = b.prepare(ip, prompt, ad, frame_num=a.frames); N = cond["num_chunks"]; sig = b.node_sigmas()
    b.reset_caches(); torch.cuda.empty_cache(); g = torch.Generator().manual_seed(sd); t0 = time.time(); outs = []
    for ci in range(N):
        z = b.chunk_noise(ci, g); x0 = None
        for i in range(len(sig)):
            x0 = b.target_x0(z, i, ci)
            if i < len(sig) - 1: s = float(sig[i + 1]); z = (1 - s) * x0 + s * b.randn_like(x0, g)
        b.commit(x0, ci); outs.append(x0.detach().float().cpu())
    gen_s = time.time() - t0; lat = torch.cat(outs, dim=1)
    with torch.no_grad(): im = b.pipe.vae.decode([lat.to(b.device)])[0].detach().float().cpu()
    V = ((im.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy(); V = (V * 255).astype("uint8")
    imageio.mimwrite(os.path.join(a.out, "%s_S%s_s%d.mp4" % (name, sc, sd)), list(V), fps=16, quality=8)
    FG = feats(V[::4]); rep[name] = dict(to_base=d(FG, FB_ref), to_fast_alone=d(FG, FF_ref), gen_s=round(gen_s, 1), frames=int(len(V)), chunks=N)
    print("[%s] S%s s%d to_base %.3f (fast_alone %.3f) to_fast %.3f gen %.0fs" % (name, sc, sd, rep[name]["to_base"], rep["fast_alone_6_12_to_base"], rep[name]["to_fast_alone"], gen_s), flush=True)
    json.dump(rep, open(rp, "w"), indent=1)
print("[C2_DONE]", rp)
