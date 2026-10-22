"""E-A pilot: teacher-corrected coarse node. Fast rolls chunk by chunk; at the chosen denoise node(s) of each (selected) chunk, Fast's x0-prediction is replaced by
Base's x0-prediction (v1 high-noise expert, CFG) computed on [prefix (clean, via I2V-style mask conditioning) + chunk (noisy at the same sigma)]. Remaining nodes are Fast's.
Displayed frames all come from Fast's final node -> no cut. Cost: Base forwards per corrected chunk = 2 (CFG) per corrected node.
usage: python -m specwm.teach_node --scene 05 --seed 1234 --nodes 0 --every 1 --out <dir> [--max-area 399360] [--self-teach]"""
import argparse, os, sys, json, time, numpy as np, torch, imageio
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--scene", required=True); ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--nodes", default="0", help="Fast node indices whose x0 is replaced by Base's (0 = sigma 1.0)")
ap.add_argument("--every", type=int, default=1, help="correct every k-th chunk"); ap.add_argument("--start", type=int, default=1, help="first chunk to correct (chunk 0 = image chunk)"); ap.add_argument("--frames", type=int, default=161)
ap.add_argument("--max-area", type=int, default=480 * 832); ap.add_argument("--self-teach", action="store_true", help="control: replace with Fast's own x0 (plumbing check)"); ap.add_argument("--out", required=True); ap.add_argument("--guide", type=float, default=5.0)
ap.add_argument("--alpha", type=float, default=1.0, help="mix: x0 <- (1-alpha) x0_fast + alpha x0_base"); ap.add_argument("--prefix-mode", default="noise", choices=["noise", "renoise"], help="latent input at prefix positions: pure noise (I2V-mask style) or the clean prefix re-noised to sigma (RePaint style)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); NODES = [int(x) for x in a.nodes.split(",")]
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding
b = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
prompt = open(S + "/p_%s.txt" % a.scene).read().strip(); act = S + "/rt2_action_" + a.scene; img = S + "/lingbot-world-v2/examples/%s/image.jpg" % a.scene
cond = b.prepare(img, prompt, act, frame_num=a.frames, max_area=a.max_area); N = cond["num_chunks"]; sig = b.node_sigmas(); CS = 3
# ---- Base (v1): high expert resident on GPU, low on CPU (unused)
# load v1 (Base) code under a distinct package name so it does not clash with v2's 'specwm'/'wan' (v2 objects are already built)
import importlib, importlib.util
os.chdir(S + "/lingbot-world"); sys.path = [S + "/lingbot-world"] + [p for p in sys.path if p != S + "/lingbot-world-v2"]
for k in [k for k in sys.modules if k == "wan" or k.startswith("wan.")]: del sys.modules[k]
spec = importlib.util.spec_from_file_location("specwm_v1", S + "/lingbot-world/specwm/__init__.py", submodule_search_locations=[S + "/lingbot-world/specwm"]); pkg = importlib.util.module_from_spec(spec); sys.modules["specwm_v1"] = pkg; spec.loader.exec_module(pkg)
V1Binding = importlib.import_module("specwm_v1.binding_v1").V1Binding
b1 = None if a.self_teach else V1Binding(S + "/lingbot-world/lingbot-world-base-cam", t5_cpu=True)
if b1 is not None: b1.pipe.low_noise_model.cpu(); b1.pipe.high_noise_model.to(b1.device); torch.cuda.empty_cache(); sig1, ts1 = b1.node_grid(70)
os.chdir(S + "/lingbot-world-v2")
vae = b.pipe.vae; dev = b.device; cond_cache = {}
def base_x0(prefix_lat, z_chunk, s):
    """Base x0-prediction for the chunk at sigma s given the clean prefix (mask conditioning: y = [mask; VAE(prefix frames + zeros)])."""
    Lp = prefix_lat.shape[1]; F = 4 * (Lp + CS - 1) + 1
    if F not in cond_cache: cond_cache[F] = b1.prepare(img, prompt, act, frame_num=F, max_area=a.max_area)
    c = dict(cond_cache[F]); h, w = c["hw"]; lat_h, lat_w = c["lat"][1:]
    with torch.no_grad():
        pref_px = vae.decode([prefix_lat.to(dev)])[0]                                   # [3, 4(Lp-1)+1, h, w]
        Pf = pref_px.shape[1]; msk = torch.ones(1, F, lat_h, lat_w, device=dev); msk[:, Pf:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1); msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
        y_lat = vae.encode([torch.concat([pref_px, torch.zeros(3, F - Pf, h, w, device=dev)], dim=1)])[0]
        c["y"] = torch.concat([msk, y_lat])
        i = int(torch.argmin((sig1 - s).abs())); t_val = float(ts1[i]); s1 = float(sig1[i]); tt = torch.full((1,), t_val, device=dev)
        want = "high" if t_val >= b1.boundary_t else "low"; models = {"high": b1.pipe.high_noise_model, "low": b1.pipe.low_noise_model}
        if next(models[want].parameters()).device.type != "cuda":   # expert swap (Fast + one Base expert resident)
            other = "low" if want == "high" else "high"; models[other].cpu(); torch.cuda.empty_cache(); models[want].to(dev)
        m = models[want]
        pref_in = torch.randn_like(prefix_lat).to(dev) if a.prefix_mode == "noise" else (1 - s1) * prefix_lat.to(dev) + s1 * torch.randn_like(prefix_lat).to(dev)
        zfull = torch.cat([pref_in, z_chunk.to(dev)], dim=1)   # prefix positions: noise (I2V-mask style) or re-noised clean prefix (RePaint style); chunk = draft's noisy state
        zin = [zfull.to(b1.param_dtype)]
        with torch.amp.autocast("cuda", dtype=b1.param_dtype):
            fc = m(zin, t=tt, context=[c["ctx"][0]], seq_len=c["seq_len"], y=[c["y"]], dit_cond_dict=c["dit_cond"])[0]
            fu = m(zin, t=tt, context=c["ctxn"], seq_len=c["seq_len"], y=[c["y"]], dit_cond_dict=c["dit_cond"])[0]
        flow = fu + a.guide * (fc - fu); x0 = (zfull - s1 * flow.float())[:, Lp:]
    return x0
b.reset_caches(); g = torch.Generator().manual_seed(a.seed); outs = []; t0 = time.time(); n_base = 0; dstat = []
for ci in range(N):
    z = b.chunk_noise(ci, g); x0 = None; do = (ci >= a.start) and ((ci - a.start) % a.every == 0)
    for i in range(len(sig)):
        x0 = b.target_x0(z, i, ci)
        if do and i in NODES and ci > 0:
            prefix = torch.cat(outs, dim=1)
            if a.self_teach: x0_t = x0
            else: x0_t = base_x0(prefix, z, float(sig[i])); n_base += 2
            dstat.append(dict(chunk=ci, node=i, rel_change=float((x0_t - x0).norm() / (x0.norm() + 1e-6)))); x0 = (1 - a.alpha) * x0 + a.alpha * x0_t
        if i < len(sig) - 1: s = float(sig[i + 1]); z = (1 - s) * x0 + s * b.randn_like(x0, g)
    b.commit(x0, ci); outs.append(x0.detach().float().cpu())
gen_s = time.time() - t0; lat = torch.cat(outs, dim=1)
with torch.no_grad(): im = vae.decode([lat.to(dev)])[0].detach().float().cpu()
v = ((im.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy(); v = (v * 255).astype("uint8")
np.save(os.path.join(a.out, "frames_all.npy"), v[::4]); imageio.mimwrite(os.path.join(a.out, "video.mp4"), list(v), fps=16, quality=8)
json.dump(dict(scene=a.scene, seed=a.seed, nodes=NODES, every=a.every, start=a.start, self_teach=a.self_teach, alpha=a.alpha, prefix_mode=a.prefix_mode, gen_s=round(gen_s, 1), base_forwards=n_base, chunks=N, rel_change=dstat), open(os.path.join(a.out, "teach.json"), "w"), indent=1)
print("[TEACH_DONE] %s seed %d nodes %s every %d: gen %.0fs, base forwards %d, mean rel change %.3f" % (a.scene, a.seed, NODES, a.every, gen_s, n_base, np.mean([d["rel_change"] for d in dstat]) if dstat else 0), flush=True)
