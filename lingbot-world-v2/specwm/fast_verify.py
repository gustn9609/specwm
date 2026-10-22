"""Draft side of Guandao's p/q verifier: Fast's own one-step reconstruction error of a clip, teacher-forced chunk by chunk.
For each blind-labelled clip: VAE-encode the first 153 frames (39 latents = 13 chunks of 3); for chunk c and Fast node i (sigma_i),
z = (1-s) x0_c + s eps_c with eps = the SAME full-clip noise tensor the Base scorer uses for that sigma (generator seeded by crc32(task|seed|sigma)),
x0_hat = Fast one-step prediction given the clean prefix in its KV cache; then commit the clean chunk. Error per latent frame = mean (x0_hat - x0)^2.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.fast_verify OUT.json [--nodes 1,2,3]  (prints the node sigmas to reuse for base_verify --sigmas)"""
import argparse, os, sys, json, time, zlib, numpy as np, torch, imageio
S = "/home/hk28965/specwm"; ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--nodes", default="1,2,3"); ap.add_argument("--frames", type=int, default=153); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--mapping", default="", help="JSON {key: {task, seed, arm}} of clips to score (default: the blind mapping)"); a = ap.parse_args()
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding
dev = torch.device("cuda:0")
b = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
sig = [float(x) for x in b.node_sigmas()]; NODES = [int(x) for x in a.nodes.split(",")]
print("[fv] node sigmas %s -> using %s" % (["%.4f" % s for s in sig], ["%.4f" % sig[i] for i in NODES]), flush=True)
T = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
B = S + "/paper/overnight/blind"; mp = json.load(open(a.mapping or (B + "/mapping.json"))); V0 = S + "/lingbot-world-v2/specwm_out/spec_v0"
clips = sorted({(v["task"], v["seed"], v["arm"]) for v in mp.values()})
if a.limit: clips = clips[:a.limit]
out = json.load(open(a.out)) if os.path.exists(a.out) else {}
cur = None; CS = 3; t0 = time.time()
for (task, seed, arm) in clips:
    k = "%s|%s|%d" % (task, arm, seed)
    if all(("%.4f" % sig[i]) in out.get(k, {}) for i in NODES): continue
    if cur != task: cond = b.prepare(T[task]["image"], "", T[task]["action"], frame_num=a.frames, max_area=480 * 832); cur = task; hw = tuple(b._chunk_shape[2:])
    fr = [np.asarray(f) for f in imageio.get_reader("%s/%s/%s_s%d/video.mp4" % (V0, task, arm, seed))][:a.frames]
    v = torch.from_numpy(np.stack(fr)).float().div(127.5).sub(1).permute(3, 0, 1, 2)
    H, W = hw[0] * 8, hw[1] * 8
    if v.shape[2:] != (H, W): v = torch.nn.functional.interpolate(v.permute(1, 0, 2, 3), size=(H, W), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
    with torch.no_grad(): x0 = b.pipe.vae.encode([v.to(dev)])[0].float()
    EPS = {}
    for i in NODES:
        s = sig[i]; g = torch.Generator(device=dev).manual_seed(zlib.crc32(("%s|%d|%s" % (task, seed, "%.4f" % s)).encode())); EPS[i] = torch.randn(x0.shape, generator=g, device=dev)
    b.reset_caches(); err = {i: [] for i in NODES}; ts = time.time()
    with torch.no_grad():
        for ci in range(x0.shape[1] // CS):
            xc = x0[:, ci * CS:(ci + 1) * CS]
            for i in NODES:
                s = sig[i]; z = (1 - s) * xc + s * EPS[i][:, ci * CS:(ci + 1) * CS]
                xh = b.target_x0(z.to(torch.bfloat16), i, ci).float()
                err[i] += ((xh - xc) ** 2).mean(dim=(0, 2, 3)).cpu().tolist()
            b.commit(xc.to(torch.bfloat16), ci)
    out[k] = {("%.4f" % sig[i]): dict(err=[round(e, 5) for e in err[i]]) for i in NODES}
    print("[fv] %-6s s%-5d %-32s %.1fs  " % (task[:6], seed, arm[:32], time.time() - ts) + " ".join("s%.2f %.4f" % (sig[i], np.mean(err[i])) for i in NODES), flush=True)
    json.dump(out, open(a.out, "w"))
print("[fv] done %.0f s" % (time.time() - t0), flush=True)
