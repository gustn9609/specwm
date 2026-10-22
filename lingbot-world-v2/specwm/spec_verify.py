"""Verify + redraw with the scene-general drift head (drift_head_train.py --all): 14B at 832x480, fast path = 2 steps (nodes 0,2 of the 4-node
schedule). At every chunk the commit pass is scored by the head (layer-20 hidden, 3x5 pooled); if the chunk logit exceeds --tau the chunk is
rejected and redrawn (--redraw resample2: same 2 steps with fresh noise, up to --tries times, keep the best-scoring; dense4: the full 4-step chunk
from the same noise). Writes lat.pt, video, loop.json (per-chunk logits, redraws, time). usage: CUDA_VISIBLE_DEVICES=g python -m specwm.spec_verify
--tasks a,b --seeds 4321 --out ROOT [--tau -3.25 --redraw resample2|dense4|none --tries 2 --fast-steps 2]"""
import argparse, os, sys, json, time, torch, torch.nn as nn, torch.nn.functional as F
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--tasks", required=True); ap.add_argument("--seeds", default="4321"); ap.add_argument("--out", required=True); ap.add_argument("--arm", default="")
ap.add_argument("--head", default=S + "/lingbot-world-v2/specwm_out/drift_head/head_L20_all.pt"); ap.add_argument("--tau", type=float, default=-3.25); ap.add_argument("--redraw", default="resample2"); ap.add_argument("--tries", type=int, default=2)
ap.add_argument("--fast-steps", type=int, default=2); ap.add_argument("--no-video", action="store_true"); ap.add_argument("--warm", type=int, default=1); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"; dev = "cuda:0"
B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(B.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 64
for blk in B.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
SIG = [float(s) for s in B.node_sigmas()]; FAST_NODES = {2: [0, 2], 3: [0, 1, 3], 4: [0, 1, 2, 3]}[a.fast_steps]; sync = torch.cuda.synchronize
# ---- drift head on the layer-20 commit features
class Head(nn.Module):
    def __init__(s, C, hid):
        super().__init__(); s.mu = nn.Parameter(torch.zeros(C), requires_grad=False); s.sd = nn.Parameter(torch.ones(C), requires_grad=False)
        s.tok = nn.Sequential(nn.Linear(C, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU()); s.pos = nn.Parameter(torch.zeros(16, hid)); s.out = nn.Sequential(nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))
    def chunk_logits(s, x): h = s.tok((x - s.mu) / s.sd) + s.pos; return s.out(h.mean(1)).squeeze(-1)
hk = torch.load(a.head, map_location="cpu"); HEAD = Head(5120, hk["hid"]).to(dev); HEAD.load_state_dict(hk["state"]); HEAD.eval(); H = {}
B.target_model.blocks[19].register_forward_hook(lambda m, i, o: H.__setitem__("h", o[0] if isinstance(o, tuple) else o))
def pool(h, f_, h2, w2):
    x = h[0].float().view(f_, h2, w2, -1).permute(3, 0, 1, 2); g = F.adaptive_avg_pool3d(x, (1, 3, 5)).reshape(x.shape[0], 15).t(); return torch.cat([x.mean((1, 2, 3))[None], g])
@torch.no_grad()
def score_commit(x, c, grid):   # run the commit pass, return the head logit for this chunk (the commit is the forward every pipeline already pays)
    B.commit(x, c); return float(HEAD.chunk_logits(pool(H["h"], *grid)[None])[0])
def gen(seed, c, k=0): return torch.Generator().manual_seed(seed * 1009 + c * 7 + 31 * k)
@torch.no_grad()
def sample(c, g, nodes):
    z = B.chunk_noise(c, g); x = None
    for j, i in enumerate(nodes):
        if j > 0: s_ = SIG[i]; z = (1 - s_) * x + s_ * B.randn_like(x, g)
        x = B.target_x0(z, i, c).float()
    return x
arm = a.arm or ("verify_%s_tau%g_t%d" % (a.redraw, a.tau, a.tries) if a.redraw != "none" else "u%d_scored" % a.fast_steps)
@torch.no_grad()
def run(task, seed, out):
    T = ALL[task]; t0 = time.time(); cond = B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); B.reset_caches(); N = cond["num_chunks"]
    grid = (3, B._chunk_shape[2] // 2, B._chunk_shape[3] // 2); prep = time.time() - t0; chunks = []; logits = []; redraws = []; t_fast = t_redraw = 0.0; tg = time.time()
    for c in range(N):
        sync(); t1 = time.time(); x = sample(c, gen(seed, c), FAST_NODES); l = score_commit(x, c, grid); sync(); t_fast += time.time() - t1; tries = 0; best = (l, x)
        if c > 0 and a.redraw != "none" and l > a.tau:
            sync(); t2 = time.time()
            for k in range(1, a.tries + 1):
                B._committed_frames -= B._chunk_size   # the ring slot of chunk c is overwritten by the next commit
                xk = sample(c, gen(seed, c, k), FAST_NODES if a.redraw == "resample2" else [0, 1, 2, 3]); lk = score_commit(xk, c, grid); tries += 1
                if lk < best[0]: best = (lk, xk)
                if lk <= a.tau: break
            if best[1] is not x:   # make sure the committed K/V is the kept chunk
                B._committed_frames -= B._chunk_size; score_commit(best[1], c, grid)
            sync(); t_redraw += time.time() - t2
        logits.append(round(best[0], 3)); redraws.append(tries); chunks.append(best[1].cpu())
    gen_s = time.time() - tg; os.makedirs(out, exist_ok=True); lat = torch.cat(chunks, 1); torch.save(lat.half(), out + "/lat.pt"); dec = 0.0
    if not a.no_video: td = time.time(); B.decode_and_save([lat.to(dev)], out); dec = time.time() - td
    R = dict(task=task, seed=seed, arm=arm, chunks=N, tau=a.tau, redraw=a.redraw, prepare_s=round(prep, 2), gen_s=round(gen_s, 2), fast_s=round(t_fast, 2), redraw_s=round(t_redraw, 2), decode_s=round(dec, 2), e2e_s=round(time.time() - t0, 2),
             logits=logits, redraws=redraws, flagged=int(sum(l > a.tau for l in logits[1:])), n_redraw_chunks=int(sum(r > 0 for r in redraws)), still_flagged=int(sum(l > a.tau for l in logits[1:])))
    json.dump(R, open(out + "/loop.json", "w"), indent=1); print("[VERIFY] %s s%d %s gen %.1fs (fast %.1f redraw %.1f) flagged-after %d/%d redrawn %d | logits %s" % (task, seed, arm, gen_s, t_fast, t_redraw, R["still_flagged"], N - 1, R["n_redraw_chunks"], " ".join("%.1f" % l for l in logits)), flush=True)
tasks = a.tasks.split(","); seeds = [int(s) for s in a.seeds.split(",")]
if a.warm: run(tasks[0], seeds[0], a.out + "/%s/%s_warm" % (tasks[0], arm))
for t in tasks:
    for s in seeds: run(t, s, a.out + "/%s/%s_s%d" % (t, arm, s))
print("[ALL_DONE]", flush=True)
