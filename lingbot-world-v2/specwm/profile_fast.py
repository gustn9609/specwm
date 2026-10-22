"""Per-chunk time breakdown of the Fast model: self-attention / cross-attention / FFN / other (norm, modulation) / VAE decode, at a given resolution and step count.
Hooks on each WanAttentionBlock's submodules with CUDA events. usage: python -m specwm.profile_fast --max-area 399360 --nodes 0,1,2,3"""
import argparse, os, time, json, torch
from .binding_v2 import V2Binding
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--action-root", required=True); ap.add_argument("--prompt-file", required=True); ap.add_argument("--scene", default="05")
ap.add_argument("--max-area", type=int, default=480 * 832); ap.add_argument("--nodes", default="0,1,2,3"); ap.add_argument("--frames", type=int, default=161); ap.add_argument("--chunks", type=int, default=8); ap.add_argument("--out", required=True)
a = ap.parse_args(); os.makedirs(os.path.dirname(a.out), exist_ok=True)
b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
prompt = open(a.prompt_file).read().strip(); cond = b.prepare(os.path.join("examples", a.scene, "image.jpg"), prompt, a.action_root, frame_num=a.frames, max_area=a.max_area)
N = min(cond["num_chunks"], a.chunks); sig = b.node_sigmas(); NODES = [int(x) for x in a.nodes.split(",")]; tm = b.target_model
acc = {"self_attn": 0.0, "cross_attn": 0.0, "ffn": 0.0}; evs = []
def hook(name):
    def pre(m, args, kwargs): e = torch.cuda.Event(enable_timing=True); e.record(); m._t0 = e
    def post(m, args, kwargs, out): e = torch.cuda.Event(enable_timing=True); e.record(); evs.append((name, m._t0, e))
    return pre, post
hs = []
for blk in tm.blocks:
    for name in acc:
        sub = getattr(blk, name, None)
        if sub is None: continue
        pre, post = hook(name); hs.append(sub.register_forward_pre_hook(pre, with_kwargs=True)); hs.append(sub.register_forward_hook(post, with_kwargs=True))
print("[profile] blocks %d, hooks %d, grid %s, chunks %d, nodes %s" % (len(tm.blocks), len(hs), tuple(cond["shape"]), N, NODES), flush=True)
def one_chunk(ci, g):
    z = b.chunk_noise(ci, g); x0 = None
    for j, i in enumerate(NODES):
        x0 = b.target_x0(z, i, ci)
        if j < len(NODES) - 1: s = float(sig[NODES[j + 1]]); z = (1 - s) * x0 + s * b.randn_like(x0, g)
    b.commit(x0, ci); return x0
g = torch.Generator().manual_seed(1234); b.reset_caches(); outs = []
one_chunk(0, g); torch.cuda.synchronize(); evs.clear()                      # warm-up chunk not counted
t_total = 0.0; t_commit = 0.0
for ci in range(1, N):
    torch.cuda.synchronize(); t0 = time.time(); z = b.chunk_noise(ci, g); x0 = None
    for j, i in enumerate(NODES):
        x0 = b.target_x0(z, i, ci)
        if j < len(NODES) - 1: s = float(sig[NODES[j + 1]]); z = (1 - s) * x0 + s * b.randn_like(x0, g)
    torch.cuda.synchronize(); t_gen = time.time() - t0; tc = time.time(); b.commit(x0, ci); torch.cuda.synchronize(); t_commit += time.time() - tc; t_total += t_gen; outs.append(x0.detach().float().cpu())
torch.cuda.synchronize()
for name, e0, e1 in evs: acc[name] += e0.elapsed_time(e1) / 1000.0
n = N - 1; lat = torch.cat(outs, dim=1); torch.cuda.synchronize(); td = time.time()
with torch.no_grad(): b.pipe.vae.decode([lat.to(b.device)])
torch.cuda.synchronize(); t_dec = (time.time() - td) / n
# attention hooks also fire during commit (t=0 call) -> subtract proportionally: commit = 1 model call, gen = len(NODES) calls
share = len(NODES) / (len(NODES) + 1); per = {k: v / n * share for k, v in acc.items()}; other = t_total / n - sum(per.values())
rep = dict(max_area=a.max_area, grid=[int(x) for x in cond["shape"]], nodes=NODES, chunks_timed=n, gen_s_per_chunk=t_total / n, commit_s_per_chunk=t_commit / n, vae_decode_s_per_chunk=t_dec,
           self_attn_s=per["self_attn"], cross_attn_s=per["cross_attn"], ffn_s=per["ffn"], other_s=other, fps_gen_only=12 / (t_total / n), fps_e2e=12 / (t_total / n + t_commit / n + t_dec))
json.dump(rep, open(a.out, "w"), indent=1); print("[PROFILE]", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rep.items()}), flush=True)
