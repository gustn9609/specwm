"""[10/14, item 2 data] Pairs for the action-edit refiner: at every chunk of a teacher-forced 4-step rollout on a synthetic key-press stream,
the TRUE-action chunk x_T and a chunk x_H generated from the SAME noise and history under a wrong action (the constant-velocity hypothesis
when the stream changes at this chunk, else a random perturbation: stop / yaw +-8 deg per frame / motion on-off). The refiner will learn
x_H (re-noised) + true action -> x_T. Output <out>/<task>_<stream>_s<seed>/{lat.pt [16,F,h,w] committed true chunks (history replay),
xh.pt list of [16,3,h,w], meta.json (rel_H, rel_T per chunk, within-eps of x_H vs x_T)}.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.edit_dump --out DIR --tasks a,b --streams tau1_s0,tau1_s1 [--seed 7] [--shard i/n]"""
import argparse, os, sys, json, time, math, random, numpy as np, torch
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--tasks", required=True); ap.add_argument("--streams", default="tau1_s0,tau1_s1"); ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--shard", default="0/1"); ap.add_argument("--eps", type=float, default=0.24); ap.add_argument("--frames", type=int, default=161); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
from specwm.act_stream import se3
from wan.utils.cam_utils import get_Ks_transformed, get_plucker_embeddings
from einops import rearrange
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"; dev = "cuda:0"; CS = 3
B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(B.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 512; _dyn.config.accumulated_recompile_limit = 8192
for blk in B.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
B.pipe.text_encoder.model.to(dev); _te = B.pipe.text_encoder
class _TE:
    def __init__(s, te): s.te = te; s.model = te.model
    def __call__(s, texts, device): return s.te(texts, dev)
B.pipe.text_encoder = _TE(_te); sig = [float(s) for s in B.node_sigmas()]; os.makedirs(a.out, exist_ok=True)
def gen(seed, c, k=0): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k)
def patch(x):
    c, f, h, w = x.shape; return x.view(c, f, 1, h // 2, 2, w // 2, 2).permute(1, 3, 5, 2, 4, 6, 0).reshape(f * (h // 2) * (w // 2), 64)
def emb_chunk(rel, Ks, h, w, lat_h, lat_w):
    f = rel.shape[0]; e = get_plucker_embeddings(rel, Ks, h, w); e = rearrange(e, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)', c1=int(h // lat_h), c2=int(w // lat_w))[None, ...]
    return rearrange(e, 'b (f h w) c -> b c f h w', f=f, h=lat_h, w=lat_w).to(B.param_dtype)
def run4(z, c, g):
    for i in range(4):
        x = B.target_x0(z, i, c).float()
        if i < 3: z = (1 - sig[i + 1]) * x + sig[i + 1] * B.randn_like(x, g)
    return x
def yawm(deg):
    cth, sth = math.cos(math.radians(deg)), math.sin(math.radians(deg)); R = torch.eye(4, device=dev); R[0, 0] = cth; R[0, 2] = sth; R[2, 0] = -sth; R[2, 2] = cth; return R
si, sn = (int(x) for x in a.shard.split("/")); cases = [(t, s) for t in a.tasks.split(",") for s in a.streams.split(",")][si::sn]; rng = random.Random(a.seed)
for task, stream in cases:
    od = "%s/%s_%s_s%d" % (a.out, task, stream, a.seed)
    if os.path.exists(od + "/meta.json"): continue
    T = ALL[task]; t0 = time.time(); os.makedirs(od, exist_ok=True)
    try:
        with torch.no_grad():
            cond = B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=a.frames, max_area=480 * 832); B.reset_caches(); N = cond["num_chunks"]
            lat_h, lat_w = B._chunk_shape[2], B._chunk_shape[3]; h, w = lat_h * 8, lat_w * 8; rel = torch.from_numpy(np.load(S + "/data/actstream/%s/rel.npy" % stream)).float().to(dev)[:N * CS]
            Ks = get_Ks_transformed(torch.from_numpy(np.load(S + "/data/actstream/%s/intrinsics.npy" % stream)).float(), height_org=480, width_org=832, height_resize=h, width_resize=w, height_final=h, width_final=w)[0].repeat(N * CS, 1).to(dev)
            E_true = [emb_chunk(rel[CS * c:CS * c + CS], Ks[CS * c:CS * c + CS], h, w, lat_h, lat_w) for c in range(N)]; B._emb_chunks = list(E_true)
            lats = []; xhs = []; meta = dict(task=task, stream=stream, seed=a.seed, chunks=[])
            for c in range(N):
                z0 = B.chunk_noise(c, gen(a.seed, c)); xT = run4(z0, c, gen(a.seed, c, 1)); tr = rel[CS * c:CS * c + CS]
                kind = "none"; rh = None
                if c >= 1:
                    hyp = rel[CS * c - 1].expand(CS, 4, 4).clone()
                    if float((hyp - tr).abs().max()) > 1e-4: rh, kind = hyp, "cv"   # the stream changes here: the deployed miss case
                    else:
                        kind = rng.choice(["stop", "yaw+8", "yaw-8", "trans"]); x = tr.clone()
                        if kind == "stop": x = torch.eye(4, device=dev).expand(CS, 4, 4).clone()
                        elif kind.startswith("yaw"): d = float(kind[3:]); x = torch.stack([x[i] @ yawm(d) for i in range(CS)])
                        else:   # motion on/off
                            if float(x[:, :3, 3].norm()) > 1e-6: x[:, :3, 3] = 0
                            else: x[:, 2, 3] = -1.0
                        rh = x
                row = dict(c=c, kind=kind)
                if rh is not None:
                    B._emb_chunks = E_true[:c] + [emb_chunk(rh, Ks[CS * c:CS * c + CS], h, w, lat_h, lat_w)] + E_true[c + 1:]
                    xH = run4(z0, c, gen(a.seed, c, 1)); B._emb_chunks = list(E_true)
                    tT = patch(xT); sg = float(tT.norm(dim=1).median()); gp = (patch(xH) - tT).norm(dim=1) / sg
                    row.update(rel_H=rh.cpu().tolist(), rel_T=tr.cpu().tolist(), within=round(float((gp <= a.eps).float().mean()), 4), gap=round(float(gp.mean()), 4)); xhs.append(xH.half().cpu())
                else: xhs.append(None)
                B.commit(xT, c); lats.append(xT.half().cpu()); meta["chunks"].append(row)
            torch.save(torch.cat(lats, 1), od + "/lat.pt"); torch.save(xhs, od + "/xh.pt"); meta["total_s"] = round(time.time() - t0, 1); json.dump(meta, open(od + "/meta.json", "w"))
        print("[ED] %s %s: %d pairs (%s) %.0fs" % (task, stream, sum(1 for r in meta["chunks"] if r["kind"] != "none"), " ".join("%s:%.0f%%" % (r["kind"], 100 * r["within"]) for r in meta["chunks"] if r["kind"] != "none"), time.time() - t0), flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print("[ED_FAIL] %s %s %r" % (task, stream, e), flush=True)
    finally: torch.cuda.empty_cache()
print("[ED_DONE]", flush=True)
