"""[10/14, flow-map post-training data] Teacher-forced 4-step rollouts of the 14B causal-fast on synthetic key-press streams: the committed
clean chunks x_T of every clip (the same tensors edit_dump.py writes as lat.pt, without the perturbed pairs). Output <out>/<task>_<stream>_s<seed>/
{lat.pt [16,F,h,w] half, meta.json}. usage: CUDA_VISIBLE_DEVICES=g python -m specwm.fm_dump --out DIR --tasks a,b --streams tau1_s0,... [--seed 7] [--shard i/n]"""
import argparse, os, sys, json, time, numpy as np, torch
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--tasks", required=True); ap.add_argument("--streams", required=True); ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--shard", default="0/1"); ap.add_argument("--frames", type=int, default=161); ap.add_argument("--area", type=int, default=480 * 832); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
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
def emb_chunk(rel, Ks, h, w, lat_h, lat_w):
    f = rel.shape[0]; e = get_plucker_embeddings(rel, Ks, h, w); e = rearrange(e, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)', c1=int(h // lat_h), c2=int(w // lat_w))[None, ...]
    return rearrange(e, 'b (f h w) c -> b c f h w', f=f, h=lat_h, w=lat_w).to(B.param_dtype)
def run4(z, c, g):
    for i in range(4):
        x = B.target_x0(z, i, c).float()
        if i < 3: z = (1 - sig[i + 1]) * x + sig[i + 1] * B.randn_like(x, g)
    return x
si, sn = (int(x) for x in a.shard.split("/")); cases = [(t, s) for t in a.tasks.split(",") for s in a.streams.split(",")][si::sn]
for task, stream in cases:
    od = "%s/%s_%s_s%d" % (a.out, task, stream, a.seed)
    if os.path.exists(od + "/meta.json"): continue
    T = ALL[task]; t0 = time.time(); os.makedirs(od, exist_ok=True)
    try:
        with torch.no_grad():
            cond = B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=a.frames, max_area=a.area); B.reset_caches(); N = cond["num_chunks"]
            lat_h, lat_w = B._chunk_shape[2], B._chunk_shape[3]; h, w = lat_h * 8, lat_w * 8; rel = torch.from_numpy(np.load(S + "/data/actstream/%s/rel.npy" % stream)).float().to(dev)[:N * CS]
            Ks = get_Ks_transformed(torch.from_numpy(np.load(S + "/data/actstream/%s/intrinsics.npy" % stream)).float(), height_org=480, width_org=832, height_resize=h, width_resize=w, height_final=h, width_final=w)[0].repeat(N * CS, 1).to(dev)
            B._emb_chunks = [emb_chunk(rel[CS * c:CS * c + CS], Ks[CS * c:CS * c + CS], h, w, lat_h, lat_w) for c in range(N)]
            lats = []
            for c in range(N):
                xT = run4(B.chunk_noise(c, gen(a.seed, c)), c, gen(a.seed, c, 1)); B.commit(xT, c); lats.append(xT.half().cpu())
            torch.save(torch.cat(lats, 1), od + "/lat.pt"); json.dump(dict(task=task, stream=stream, seed=a.seed, area=a.area, frames=a.frames, chunks=[dict(c=c, kind="none") for c in range(N)], total_s=round(time.time() - t0, 1)), open(od + "/meta.json", "w"))
        print("[FD] %s %s %.0fs" % (task, stream, time.time() - t0), flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print("[FD_FAIL] %s %s %r" % (task, stream, e), flush=True)
    finally: torch.cuda.empty_cache()
print("[FD_DONE]", flush=True)
