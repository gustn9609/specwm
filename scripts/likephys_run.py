#!/usr/bin/env python3
"""LikePhys benchmark with LingBot Base as the likelihood scorer (one process, weights loaded once).
Protocol (LikePhys): 10 uniform timesteps over the sampler grid, flow/noise MSE, cond only; static-camera pose; prompt per scenario; first frame of each video as image cond.
Videos: 512x512, 60 f @30fps -> use 57 frames (4n+1), max_area 184320. PPE per scenario/variation = fraction of (valid, invalid) pairs (within subgroup) with loss_valid > loss_invalid.
usage: python -m ... (run from lingbot-world with PYTHONPATH) --scenarios ball_drop,... --per-sub K --out json"""
import argparse, json, os, glob, time, numpy as np, torch, imageio
PROMPT = {"ball_drop": "ball dropping and colliding with the ground, in empty background", "ball_collision": "two balls colliding with each other", "pendulum": "a pendulum swinging", "block_slide": "a block sliding on a slope", "fluid": "a droplet falling", "faucet": "fluid flowing from a faucet", "cloth_drape": "a piece of cloth dropping to the obstacle on the ground", "flag": "a piece of cloth waving in the wind", "river": "fluid flowing in a tank with obstacles", "shadow_camera": "camera moving around an object", "pyramid": "a cube crash into a pile of spheres"}
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True); ap.add_argument("--scenarios", default=",".join(PROMPT)); ap.add_argument("--per-sub", type=int, default=99); ap.add_argument("--subgroups", type=int, default=10)
ap.add_argument("--frames", type=int, default=57); ap.add_argument("--max-area", type=int, default=184320); ap.add_argument("--n-t", type=int, default=10); ap.add_argument("--out", required=True); ap.add_argument("--action-dir", required=True)
a = ap.parse_args(); S = "/home/hk28965/specwm"
from specwm.binding_v1 import V1Binding
b = V1Binding(a.ckpt, t5_cpu=True); models = {"high": b.pipe.high_noise_model, "low": b.pipe.low_noise_model}; models["high"].cpu(); models["low"].to(b.device); torch.cuda.empty_cache()
sig_all, ts_all = b.node_grid(70); ids = [int(i * len(sig_all) / a.n_t) for i in range(a.n_t)]
res = json.load(open(a.out)) if os.path.exists(a.out) else {}
def score(video, prompt):
    import cv2
    fr = [np.asarray(f) for f in imageio.get_reader(video)]
    nf = min(a.frames, ((len(fr) - 1) // 4) * 4 + 1); fr = fr[:nf]; tmp = a.out + ".first.png"; imageio.imwrite(tmp, fr[0])   # 4n+1 로 맞춤 (block_slide 56f -> 53)
    cond = b.prepare(tmp, prompt, a.action_dir, frame_num=nf, max_area=a.max_area); hw = tuple(cond["hw"]) if "hw" in cond else fr[0].shape[:2]
    if fr[0].shape[:2] != tuple(hw): fr = [cv2.resize(f, (hw[1], hw[0]), interpolation=cv2.INTER_AREA) for f in fr]
    x = torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float() / 127.5 - 1.0
    with torch.no_grad(): x0 = b.pipe.vae.encode([x.to(b.device)])[0].float()
    g = torch.Generator().manual_seed(0); fl = []
    with torch.no_grad():
        for i in ids:
            s, t_val = float(sig_all[i]), float(ts_all[i]); use_high = t_val >= b.boundary_t; mdl = models["high"] if use_high else models["low"]
            if next(mdl.parameters()).device.type != "cuda":
                (models["low"] if use_high else models["high"]).cpu(); mdl.to(b.device); torch.cuda.empty_cache()
            noise = torch.randn(x0.shape, generator=g).to(b.device); z = (1 - s) * x0 + s * noise; tt = torch.full((1,), t_val, device=b.device)
            with torch.amp.autocast("cuda", dtype=b.param_dtype):
                fc = mdl([z.to(b.param_dtype)], t=tt, context=[cond["ctx"][0]], seq_len=cond["seq_len"], y=[cond["y"]], dit_cond_dict=cond["dit_cond"])[0]
            fl.append(float(((fc.float() - (noise - x0)) ** 2).mean()))
    return fl
for sc in a.scenarios.split(","):
    d = a.data + "/%s_videos" % sc; prompt = PROMPT[sc]
    for sub in sorted(glob.glob(d + "/subgroup_*"))[: a.subgroups]:
        vids = sorted(glob.glob(sub + "/*.mp4")); vids = [v for v in vids if os.path.basename(v).startswith("valid")][:1] + [v for v in vids if not os.path.basename(v).startswith("valid")][: a.per_sub]
        for v in vids:
            key = os.path.relpath(v, a.data)
            if key in res: continue
            t0 = time.time(); fl = score(v, prompt); res[key] = dict(scenario=sc, subgroup=os.path.basename(sub), variant=("valid" if os.path.basename(v).startswith("valid") else os.path.basename(v).rsplit("_", 1)[0]), loss=float(np.mean(fl)), loss_array=fl)
            json.dump(res, open(a.out, "w")); print("[%s] %s loss %.4f (%.0fs)" % (sc, key, res[key]["loss"], time.time() - t0), flush=True)
# PPE per scenario / variant
out = {}
for sc in a.scenarios.split(","):
    rs = [r for r in res.values() if r["scenario"] == sc]; subs = sorted({r["subgroup"] for r in rs}); vars_ = sorted({r["variant"] for r in rs if r["variant"] != "valid"}); pp = {}
    for var in vars_:
        pairs = []
        for sg in subs:
            va = [r["loss"] for r in rs if r["subgroup"] == sg and r["variant"] == "valid"]; iv = [r["loss"] for r in rs if r["subgroup"] == sg and r["variant"] == var]
            pairs += [(x, y) for x in va for y in iv]
        if pairs: pp[var] = dict(ppe=float(np.mean([x > y for x, y in pairs])), n=len(pairs))
    out[sc] = dict(variants=pp, ppe_mean=float(np.mean([v["ppe"] for v in pp.values()])) if pp else None)
json.dump(dict(per_video=res, ppe=out), open(a.out, "w"), indent=1)
for sc, v in out.items(): print("PPE %-14s mean %.3f  %s" % (sc, v["ppe_mean"] if v["ppe_mean"] is not None else -1, {k: round(x["ppe"], 2) for k, x in v["variants"].items()}), flush=True)
print("[LIKEPHYS_DONE]")
