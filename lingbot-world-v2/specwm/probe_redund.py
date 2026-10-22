"""[10/15] Where is the redundancy a cheaper 14B forward could exploit? Hooks the causal-fast 14B's block outputs during the solo 4-step
teacher rollout and measures, per token (one DiT token = 2x2 latent patch of one latent frame), the relative change of the hidden state
  (a) across consecutive denoising steps of the same chunk (headroom for cross-step feature caching / step-asymmetric depth),
  (b) across consecutive chunks at the same step, same token index (headroom for cross-chunk reuse) and at the chunk boundary
      (last latent frame of chunk c-1 vs first frame of chunk c),
  (c) the block's residual update |block(h) - h| / |h| (headroom for per-token layer skipping).
Aggregated over chunks 3.. of each clip. usage: CUDA_VISIBLE_DEVICES=g python -m specwm.probe_redund --tasks easy_lake,billiard_break --layers 1,5,10,20,30,40"""
import argparse, os, sys, json, time, torch
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--tasks", default="easy_lake,billiard_break,hydraulic_press_can_crush"); ap.add_argument("--layers", default=",".join(str(i) for i in range(1, 41)))
ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--frames", type=int, default=161); ap.add_argument("--area", type=int, default=270336); ap.add_argument("--nodes", default="0,250,500,750")
ap.add_argument("--wq", default="fp8"); ap.add_argument("--window", type=int, default=18); ap.add_argument("--out", default=S + "/lingbot-world-v2/specwm_out/probe_redund.json")
a = ap.parse_args(); sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
dev = "cuda:0"
def log(*m): print("[probe %s]" % time.strftime("%H:%M:%S"), *m, flush=True)
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
NODES = tuple(int(v) for v in a.nodes.split(","))
Tg = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=a.window, swap_models=False, build_draft=False, timesteps_index=NODES); tm = Tg.target_model
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True)
if a.wq == "fp8": FP.fp8_swap(tm, True, keep=False)
SIG = [float(s) for s in Tg.node_sigmas()]; LAY = [int(v) for v in a.layers.split(",")]; log("nodes", NODES, "sigmas", [round(s, 3) for s in SIG], "layers", LAY)
CAP = {}; ON = [False]
def mk(L):
    def pre(m, args, kwargs): CAP[("in", L)] = args[0].detach() if ON[0] else None
    def post(m, args, out): CAP[("out", L)] = (out[0] if isinstance(out, tuple) else out).detach() if ON[0] else None
    return pre, post
for L in LAY:
    pre, post = mk(L); tm.blocks[L - 1].register_forward_pre_hook(pre, with_kwargs=True); tm.blocks[L - 1].register_forward_hook(post)
def gen(seed, c, k=0): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k * 100003)
def commit(x, c):
    fs = Tg._frame_seqlen; L = Tg._chunk_size * fs; FP.ACTIVE["a"] = dict(idx=torch.arange(L - fs, L, device=dev), n=fs, L=L)
    try: Tg.commit(x, c)
    finally: FP.ACTIVE["a"] = None
def tok(h):  # [1, L, C] -> [L, C] float
    return h[0].float()
def relrow(x, y): return ((x - y).norm(dim=-1) / y.norm(dim=-1).clamp_min(1e-6))
TH = [0.1, 0.2, 0.3, 0.5]
def summ(r):  # per-token relative changes -> dict
    r = r.flatten(); return dict(n=r.numel(), med=float(r.median()), mean=float(r.mean()), frac=[float((r < t).float().mean()) for t in TH])
def merge(acc, key, r):
    if key not in acc: acc[key] = []
    acc[key].append(r.flatten().cpu())
RES = {}
with torch.no_grad():
    for task in a.tasks.split(","):
        T = ALL[task]; cond = Tg.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=a.frames, max_area=a.area); Tg.reset_caches(); N = cond["num_chunks"]
        g = gen(a.seed, 0); prev = None; acc = {}; fs = Tg._frame_seqlen; t0 = time.time()
        for c in range(N):
            x = None; cur = {}
            for j, i in enumerate(range(len(NODES))):
                z = Tg.chunk_noise(c, g) if j == 0 else (1 - SIG[i]) * x + SIG[i] * Tg.randn_like(x, g)
                ON[0] = True; x = Tg.target_x0(z, i, c).float(); ON[0] = False
                cur[j] = {L: (tok(CAP[("in", L)]).half(), tok(CAP[("out", L)]).half()) for L in LAY}
            commit(x, c)
            if c >= 3:
                for L in LAY:
                    for j in range(len(NODES)):
                        hin, hout = cur[j][L]; hin = hin.float(); hout = hout.float(); r = hout - hin
                        def cosw(x, y): return float(torch.nn.functional.cosine_similarity(x.flatten()[None], y.flatten()[None]))
                        def cost(x, y): return torch.nn.functional.cosine_similarity(x, y, dim=-1)
                        if prev is not None:
                            pin, pout = prev[j][L]; pin = pin.float(); pout = pout.float(); pr = pout - pin
                            merge(acc, ("in_cos_chunk", L, j), torch.tensor([cosw(hin, pin)]))                 # X-Cache gate: block-input cosine c-1 vs c
                            merge(acc, ("res_cos_chunk", L, j), torch.tensor([cosw(r, pr)]))                  # residual cosine (whole chunk)
                            merge(acc, ("res_tokcos_chunk", L, j), cost(r, pr))                               # per-token residual cosine
                            merge(acc, ("reuse_err_chunk", L, j), (r - pr).norm(dim=-1) / hout.norm(dim=-1).clamp_min(1e-6))   # per-token output error if c-1 residual reused
                        # spatial predictability of the residual: vs the mean of the 4 in-frame neighbours (same chunk, same step)
                        f_, h_, w_ = Tg._chunk_size, Tg._chunk_shape[2] // 2, Tg._chunk_shape[3] // 2
                        rg = r.view(f_, h_, w_, -1); nb = torch.zeros_like(rg); cnt = torch.zeros(f_, h_, w_, 1, device=rg.device)
                        nb[:, 1:] += rg[:, :-1]; cnt[:, 1:] += 1; nb[:, :-1] += rg[:, 1:]; cnt[:, :-1] += 1; nb[:, :, 1:] += rg[:, :, :-1]; cnt[:, :, 1:] += 1; nb[:, :, :-1] += rg[:, :, 1:]; cnt[:, :, :-1] += 1
                        nb = (nb / cnt).view_as(r)
                        merge(acc, ("res_nbrcos", L, j), cost(r, nb)); merge(acc, ("nbr_err", L, j), (r - nb).norm(dim=-1) / hout.norm(dim=-1).clamp_min(1e-6))
                        if j > 0:
                            qin, qout = cur[j - 1][L]; qr = qout.float() - qin.float()
                            merge(acc, ("in_cos_step", L, j), torch.tensor([cosw(hin, qin.float())]))
                            merge(acc, ("res_cos_step", L, j), torch.tensor([cosw(r, qr)]))
                            merge(acc, ("res_tokcos_step", L, j), cost(r, qr))
                            merge(acc, ("reuse_err_step", L, j), (r - qr).norm(dim=-1) / hout.norm(dim=-1).clamp_min(1e-6))
            prev = cur
        RES[task] = {"%s|L%d|s%d" % k: summ(torch.cat(v)) for k, v in acc.items()}; RES[task]["_meta"] = dict(N=N, tokens_per_chunk=int(Tg._chunk_size * fs), frame_tokens=int(fs), clip_s=round(time.time() - t0, 1))
        json.dump(RES, open(a.out, "w"), indent=0); log("clip", task, "done in %.0f s" % (time.time() - t0))
# ---- report ----
def tab(kind, title, fmt="%.2f"):
    print("\n### %s" % title); tasks = list(RES); J = len(NODES)
    print("| layer | " + " | ".join("%s s%d" % (t[:8], j) for t in tasks for j in range(J)) + " |"); print("|---|" + "---|" * (len(tasks) * J))
    for L in LAY:
        cells = []
        for t in tasks:
            for j in range(J):
                r = RES[t].get("%s|L%d|s%d" % (kind, L, j)); cells.append((fmt % r["med"]) if r else "–")
        print("| %d | %s |" % (L, " | ".join(cells)))
for kind, title in [("in_cos_chunk", "block-input cosine, chunk c-1 vs c (X-Cache gate; median over chunks)"), ("res_cos_chunk", "residual cosine, chunk c-1 vs c (whole chunk)"), ("res_tokcos_chunk", "per-token residual cosine, chunk c-1 vs c (median token)"), ("reuse_err_chunk", "per-token output error if the previous chunk's residual is reused (median token)"),
                    ("in_cos_step", "block-input cosine, step j-1 vs j"), ("res_cos_step", "residual cosine, step j-1 vs j"), ("res_tokcos_step", "per-token residual cosine, step j-1 vs j"), ("reuse_err_step", "per-token output error if the previous step's residual is reused"), ("res_nbrcos", "per-token residual cosine vs the mean of its 4 in-frame neighbours (same chunk, same step)"), ("nbr_err", "per-token output error if the residual is replaced by the neighbour mean")]:
    tab(kind, title)
log("DONE", a.out)
