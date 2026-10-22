"""Guandao's whiteboard, offline: Base (target) scores a WHOLE drafted clip with ONE denoising step.
For every blind-labelled clip: VAE-encode the first 153 frames -> x0 [16,39,h,w]; for each sigma, z = (1-s) x0 + s eps (eps fixed per task/seed/sigma,
shared by all methods of a case), one Base forward (conditional, empty prompt, start image + camera, no CFG) -> x0_hat = z - s * v;
record per-latent-frame reconstruction error |x0_hat - x0|^2 (mean over c,h,w). Experts are swapped once per sigma group (one expert on the GPU).
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.base_verify OUT.json [--sigmas 0.3,0.5,0.7,0.95] [--limit N]"""
import argparse, os, sys, json, time, zlib, numpy as np, torch, imageio
S = "/home/hk28965/specwm"; ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--sigmas", default="0.3,0.5,0.7,0.95"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--mapping", default="", help="JSON {key: {task, seed, arm}} of clips to score (default: the blind mapping)")
ap.add_argument("--frames", type=int, default=153); ap.add_argument("--area", type=int, default=480 * 832, help="score at this resolution (184320 = plan res 576x304)"); ap.add_argument("--shard", default="0/1", help="i/n: this process handles clips i, i+n, ..."); a = ap.parse_args()
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world"); os.chdir(S + "/lingbot-world")
from specwm.binding_v1 import V1Binding
dev = torch.device("cuda:0"); b1 = V1Binding(os.environ.get("SPECWM_BASE_CKPT") or (S + "/lingbot-world/lingbot-world-base-cam"), t5_cpu=True, device=str(dev))
EXP = {"high": b1.pipe.high_noise_model, "low": b1.pipe.low_noise_model}
for m in EXP.values(): m.to("cpu"); m.eval()
torch.cuda.empty_cache(); print("[bv] boundary t = %.0f" % b1.boundary_t, flush=True)
T = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
B = S + "/paper/overnight/blind"; mp = json.load(open(a.mapping or (B + "/mapping.json"))); V0 = S + "/lingbot-world-v2/specwm_out/spec_v0"
clips = sorted({(v["task"], v["seed"], v["arm"]) for v in mp.values()})
_si, _sn = (int(x) for x in a.shard.split("/")); clips = clips[_si::_sn]
if a.limit: clips = clips[:a.limit]
out = json.load(open(a.out)) if os.path.exists(a.out) else {}
COND = {}
def cond(task):
    if task not in COND: COND[task] = b1.prepare(T[task]["image"], "", T[task]["action"], frame_num=a.frames, max_area=a.area)
    return COND[task]
LAT = {}
t0 = time.time()
for (task, seed, arm) in clips:   # 1) encode every clip once (latents kept on CPU)
    k = "%s|%s|%d" % (task, arm, seed)
    fr = [np.asarray(f) for f in imageio.get_reader("%s/%s/%s_s%d/video.mp4" % (V0, task, arm, seed))][:a.frames]
    c = cond(task); h, w = c["hw"]
    v = torch.from_numpy(np.stack(fr)).float().div(127.5).sub(1).permute(3, 0, 1, 2)
    if v.shape[2:] != (h, w): v = torch.nn.functional.interpolate(v.permute(1, 0, 2, 3), size=(h, w), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
    with torch.no_grad(): LAT[k] = b1.pipe.vae.encode([v.to(dev)])[0].float().cpu()
print("[bv] encoded %d clips in %.0f s, latent %s" % (len(LAT), time.time() - t0, tuple(next(iter(LAT.values())).shape)), flush=True)
sig = [float(x) for x in a.sigmas.split(",")]
for ex in ("low", "high"):   # 2) one expert on the GPU at a time
    ss = [s for s in sig if (1000 * s >= b1.boundary_t) == (ex == "high")]
    if not ss: continue
    m = EXP[ex].to(dev); torch.cuda.empty_cache()
    for s in ss:
        for (task, seed, arm) in clips:
            k = "%s|%s|%d" % (task, arm, seed)
            if str(s) in out.get(k, {}): continue
            c = cond(task); x0 = LAT[k].to(dev)
            g = torch.Generator(device=dev).manual_seed(zlib.crc32(("%s|%d|%s" % (task, seed, s)).encode()))
            eps = torch.randn(x0.shape, generator=g, device=dev); z = (1 - s) * x0 + s * eps
            tt = torch.full((1,), 1000.0 * s, device=dev); ts = time.time()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                v_ = m([z.to(torch.bfloat16)], t=tt, context=[c["ctx"][0]], seq_len=c["seq_len"], y=[c["y"]], dit_cond_dict=c["dit_cond"])[0].float()
            xh = z - s * v_; err = ((xh - x0) ** 2).mean(dim=(0, 2, 3)).cpu().tolist()
            out.setdefault(k, {})[str(s)] = dict(err=[round(e, 5) for e in err], sec=round(time.time() - ts, 2))
            print("[bv] %-6s s%-5d %-32s sigma %.2f %-4s %.1fs  err mean %.4f max %.4f" % (task[:6], seed, arm[:32], s, ex, time.time() - ts, np.mean(err), max(err)), flush=True)
        json.dump(out, open(a.out, "w"))
    EXP[ex].to("cpu"); torch.cuda.empty_cache()
json.dump(out, open(a.out, "w")); print("[bv] done %.0f s" % (time.time() - t0), flush=True)
