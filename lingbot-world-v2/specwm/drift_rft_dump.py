"""Data for the drift-reduction fine-tune (track A): 14B 2-step rollouts at 576x320 where every chunk is drawn K times from the same context and
scored with the drift head at its commit pass; the lowest-scoring candidate is kept (committed) so the trajectory stays as clean as the model can
make it. Saves per clip: lat.pt (kept chunks), x0.pt {node: [N,16,3,h,w]} for the kept candidate, meta.json (per-chunk logits of all candidates,
kept index, seeds). Chunk noise of candidate k: gen(seed, c, k). usage: CUDA_VISIBLE_DEVICES=g python -m specwm.drift_rft_dump --tasks @list --seeds 1,2,3 --out ROOT [--k 4]"""
import argparse, os, sys, json, time, torch, torch.nn as nn, torch.nn.functional as F
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--tasks", required=True); ap.add_argument("--seeds", default="1,2,3"); ap.add_argument("--out", required=True); ap.add_argument("--k", type=int, default=4)
ap.add_argument("--head", default=S + "/lingbot-world-v2/specwm_out/drift_head/head_L20_all.pt"); ap.add_argument("--max-area", type=int, default=175104); ap.add_argument("--shard", default="0/1"); ap.add_argument("--losers", action="store_true", help="second pass: for existing clips regenerate the WORST candidate of every chunk under the kept context and save x0_loser.pt {node: [N,16,3,h,w]}"); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"; dev = "cuda:0"
B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 500))
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(B.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 64
for blk in B.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
SIG = [float(s) for s in B.node_sigmas()]
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
def score_commit(x, c, grid): B.commit(x, c); return float(HEAD.chunk_logits(pool(H["h"], *grid)[None])[0])
def gen(seed, c, k): return torch.Generator().manual_seed(seed * 1009 + c * 7 + 31 * k)
@torch.no_grad()
def sample(c, g):
    z = B.chunk_noise(c, g); xs = []
    for i in range(len(SIG)):
        if i > 0: s_ = SIG[i]; z = (1 - s_) * xs[-1] + s_ * B.randn_like(xs[-1], g)
        xs.append(B.target_x0(z, i, c).float())
    return xs
tasks = open(a.tasks[1:]).read().split() if a.tasks.startswith("@") else a.tasks.split(","); seeds = [int(s) for s in a.seeds.split(",")]; si, sn = [int(v) for v in a.shard.split("/")]
jobs = [(t, s) for t in tasks for s in seeds][si::sn]; done = 0
for task, seed in jobs:
    out = "%s/%s_s%d" % (a.out, task, seed)
    if a.losers:
        if not os.path.exists(out + "/meta.json") or os.path.exists(out + "/x0_loser.pt"): continue
        meta = json.load(open(out + "/meta.json")); T = ALL[task]; L = torch.load(out + "/lat.pt").float().to(dev); XL = {0: [], 1: []}
        with torch.no_grad():
            B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=meta["max_area"]); B.reset_caches(); B.commit(L[:, :3], 0)
            for c in range(1, meta["chunks"]):
                cm = meta["chunks_meta"][c]; kw = int(max(range(len(cm["logits"])), key=lambda i: cm["logits"][i])); xs = sample(c, gen(seed, c, kw)); XL[0].append(xs[0].half().cpu()); XL[1].append(xs[1].half().cpu()); B.commit(L[:, 3 * c:3 * c + 3], c)
        torch.save({k: torch.stack(v) for k, v in XL.items()}, out + "/x0_loser.pt"); done += 1; print("[RFT losers] %s s%d" % (task, seed), flush=True); continue
    if os.path.exists(out + "/meta.json"): continue
    T = ALL[task]; t0 = time.time()
    with torch.no_grad():
        cond = B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=a.max_area); B.reset_caches(); N = cond["num_chunks"]; grid = (3, B._chunk_shape[2] // 2, B._chunk_shape[3] // 2)
        kept = []; X0 = {0: [], 1: []}; meta = []
        for c in range(N):
            K = 1 if c == 0 else a.k; cands = []
            for k in range(K):
                xs = sample(c, gen(seed, c, k)); l = score_commit(xs[-1], c, grid); cands.append((l, k, xs))
                if k < K - 1: B._committed_frames -= B._chunk_size
            best = min(cands, key=lambda t: t[0])
            if best[1] != cands[-1][1]: B._committed_frames -= B._chunk_size; score_commit(best[2][-1], c, grid)   # re-commit the kept candidate
            kept.append(best[2][-1].half().cpu()); X0[0].append(best[2][0].half().cpu()); X0[1].append(best[2][1].half().cpu()); meta.append(dict(logits=[round(t[0], 3) for t in cands], kept=best[1]))
    os.makedirs(out, exist_ok=True); torch.save(torch.cat(kept, 1), out + "/lat.pt"); torch.save({k: torch.stack(v) for k, v in X0.items()}, out + "/x0.pt")
    json.dump(dict(task=task, seed=seed, chunks=N, k=a.k, max_area=a.max_area, sigmas=SIG, chunks_meta=meta, gen_s=round(time.time() - t0, 1)), open(out + "/meta.json", "w"))
    done += 1; print("[RFT] %s s%d %.0fs | kept logits %s | min-of-K gain %.2f" % (task, seed, time.time() - t0, " ".join("%.1f" % m["logits"][m["kept"]] for m in meta[1:]), sum(max(m["logits"]) - min(m["logits"]) for m in meta[1:]) / max(1, N - 1)), flush=True)
print("[RFT_DONE] %d" % done, flush=True)
