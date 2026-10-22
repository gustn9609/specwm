"""[10/14, proposal 5 go/no-go] Perceptual acceptability of the 14B's intermediate steps. Teacher-forced 4-step rollout (stock 14B, StEvo
path, as spec_early uniform:4); at every chunk the step-1/2/3 x0 estimates and the final 4-step chunk are each DECODED from the same VAE
cache state (snapshot / restore) and saved as a 4-frame strip (frames 0, 4, 8, 11 of the chunk) next to the stats: true within-eps of the
estimate vs the final (token gap), and the gap head's predicted within-eps fraction (the chunk-wise rule's statistic). The strips are then
judged pairwise by the VLM (scripts/percep_judge.py). Output <out>/<task>_s<seed>/{c<cc>_k<k>.jpg, stats.json}.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.percep_dump --out DIR [--tasks a,b] [--seeds 7] [--shard i/n]"""
import argparse, os, sys, json, time, numpy as np, torch, cv2
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--seeds", default="7"); ap.add_argument("--shard", default="0/1")
ap.add_argument("--tasks", default=open(S + "/lingbot-world-v2/specwm_out/distill_heldout12.txt").read().strip()); ap.add_argument("--eps", type=float, default=0.24); ap.add_argument("--frames", type=int, default=161); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
from specwm.early_exit import EarlyExit
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"; dev = "cuda:0"; sync = torch.cuda.synchronize
si, sn = (int(x) for x in a.shard.split("/")); tasks = a.tasks.split(",")[si::sn]; seeds = [int(s) for s in a.seeds.split(",")]
B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(B.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 512; _dyn.config.accumulated_recompile_limit = 8192
for blk in B.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
B.pipe.text_encoder.model.to(dev); _te = B.pipe.text_encoder
class _TE:
    def __init__(s, te): s.te = te; s.model = te.model
    def __call__(s, texts, device): return s.te(texts, dev)
B.pipe.text_encoder = _TE(_te); EE = EarlyExit(B, eps=a.eps, smooth=True); sig = [float(s) for s in B.node_sigmas()]; VAE = B.pipe.vae; VM = VAE.model; os.makedirs(a.out, exist_ok=True)
def gen(seed, c, k=0): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k)
def patch(x):
    c, f, h, w = x.shape; return x.view(c, f, 1, h // 2, 2, w // 2, 2).permute(1, 3, 5, 2, 4, 6, 0).reshape(f * (h // 2) * (w // 2), 64)
def vae_snapshot(): return [t.clone() if torch.is_tensor(t) else t for t in VM._feat_map]
def vae_restore(snap): VM._feat_map = [t.clone() if torch.is_tensor(t) else t for t in snap]
@torch.no_grad()
def decode(z):
    with torch.amp.autocast("cuda", dtype=VAE.dtype):
        sc = VAE.scale; zz = z.unsqueeze(0).to(VAE.dtype) / sc[1].view(1, -1, 1, 1, 1) + sc[0].view(1, -1, 1, 1, 1); x = VM.conv2(zz); outs = []
        for i in range(x.shape[2]):
            VM._conv_idx = [0]; o = VM.decoder(x[:, :, i:i + 1], feat_cache=VM._feat_map, feat_idx=VM._conv_idx).float().clamp_(-1, 1); outs.append(((o[0] + 1) * 127.5).to(torch.uint8).permute(1, 2, 3, 0))
    return torch.cat(outs, 0).cpu().numpy()
def strip(fr):
    idx = [0, min(4, len(fr) - 1), min(8, len(fr) - 1), len(fr) - 1]; return np.concatenate([fr[i] for i in idx], 1)
for task in tasks:
    for seed in seeds:
        od = "%s/%s_s%d" % (a.out, task, seed)
        if os.path.exists(od + "/stats.json"): continue
        T = ALL[task]; os.makedirs(od, exist_ok=True); t0 = time.time(); R = dict(task=task, seed=seed, chunks=[])
        with torch.no_grad():
            cond = B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=a.frames, max_area=480 * 832); B.reset_caches(); VM.clear_cache(); N = cond["num_chunks"]
            for c in range(N):
                z = B.chunk_noise(c, gen(seed, c)); g = gen(seed, c, 1); shape = tuple(z.shape); f_, h2, w2 = shape[1], shape[2] // 2, shape[3] // 2; xs = []; preds = []
                for i in range(4):
                    x = B.target_x0(z, i, c).float(); xs.append(x)
                    if i < 3: preds.append(float((EE.predict(f_, h2, w2).flatten() <= a.eps).float().mean())); z = (1 - sig[i + 1]) * x + sig[i + 1] * B.randn_like(x, g)
                tT = patch(xs[3]); sg = float(tT.norm(dim=1).median()); row = dict(c=c, head_frac=[round(p, 4) for p in preds], within=[], gap=[])
                snap = vae_snapshot()
                for k in range(3):
                    gp = (patch(xs[k]) - tT).norm(dim=1) / sg; row["within"].append(round(float((gp <= a.eps).float().mean()), 4)); row["gap"].append(round(float(gp.mean()), 4))
                    fr = decode(xs[k]); cv2.imwrite("%s/c%02d_k%d.jpg" % (od, c, k + 1), cv2.cvtColor(strip(fr), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92]); vae_restore(snap)
                fr = decode(xs[3]); cv2.imwrite("%s/c%02d_k4.jpg" % (od, c), cv2.cvtColor(strip(fr), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])   # final: cache state advances
                B.commit(xs[3], c); R["chunks"].append(row)
                print("[PD] %s c%02d head %s within %s" % (task, c, row["head_frac"], row["within"]), flush=True)
        R["total_s"] = round(time.time() - t0, 1); json.dump(R, open(od + "/stats.json", "w")); print("[PD] %s s%d %.0fs" % (task, seed, time.time() - t0), flush=True); torch.cuda.empty_cache()
print("[PD_DONE]", flush=True)
