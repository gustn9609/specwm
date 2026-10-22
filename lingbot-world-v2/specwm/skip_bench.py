"""[10/15, step (a) of the verified-skip plan] Block-skipping drafts on the causal-fast 14B: during a denoising step, blocks in the skip set are
replaced by "K/V projection only + previous residual" (the block's K/V for this chunk are still computed from its input and written to the ring
cache, so later chunks see a correct context; attention, cross-attention and FFN are skipped and the block's residual from --src is added:
  src=chunk : this block's residual at the same step in the previous chunk (X-Cache / C3ache style)
  src=step  : this block's residual at the previous step of the same chunk (TeaCache style; step 0 can't use it)
Modes: tf   = teacher-forced: per chunk the skipped pass and the full pass run from the same state, the full pass is committed; reports the
              per-token relative gap of the skipped output to the full one (and, for scale, the gap of the 2-step draft to the 4-step teacher
              is in facts §35 / the acceptance tables).
       free = the skipped output is committed and shown (deployment); writes lat.pt / spec.mp4 and timing per chunk.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.skip_bench --out DIR --mode tf --skip "s0:6-39;s1:6-39" --src chunk --nodes 0,500 --use 0,1 --tasks a,b"""
import argparse, os, sys, json, time, math, torch
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--mode", default="tf", choices=["tf", "free"])
ap.add_argument("--skip", default="", help='per-step skip sets, e.g. "s0:6-39;s1:6-39" (block indices 0-based, inclusive ranges, comma lists)'); ap.add_argument("--src", default="chunk", choices=["chunk", "step"])
ap.add_argument("--warm", type=int, default=3, help="chunks computed fully before skipping starts"); ap.add_argument("--kv-only", type=int, default=1, help="1: skipped blocks still write their K/V; 0: leave the cache slot stale")
ap.add_argument("--tasks", default="easy_lake"); ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--frames", type=int, default=161); ap.add_argument("--area", type=int, default=270336)
ap.add_argument("--nodes", default="0,500"); ap.add_argument("--use", default="0,1"); ap.add_argument("--wq", default="fp8"); ap.add_argument("--window", type=int, default=18); ap.add_argument("--compile", type=int, default=1); ap.add_argument("--video", type=int, default=1)
a = ap.parse_args(); a.out = os.path.abspath(a.out); os.makedirs(a.out, exist_ok=True)
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
dev = "cuda:0"; GiB = 2 ** 30; sync = torch.cuda.synchronize; now = time.time
def log(*m): print("[skip %s]" % time.strftime("%H:%M:%S"), *m, flush=True)
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
from wan.modules import model_fast as M
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
NODES = tuple(int(v) for v in a.nodes.split(",")); USE = [int(v) for v in a.use.split(",")]
Tg = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=a.window, swap_models=False, build_draft=False, timesteps_index=NODES); tm = Tg.target_model
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True)
if a.wq == "fp8": FP.fp8_swap(tm, True, keep=False)
SIG = [float(s) for s in Tg.node_sigmas()]
def parse_skip(spec):
    out = {}
    for part in [p for p in spec.split(";") if p.strip()]:
        s_, ls = part.split(":"); j = int(s_.strip()[1:]); st = set()
        for r in ls.split(","):
            r = r.strip()
            if "-" in r: lo, hi = r.split("-"); st.update(range(int(lo), int(hi) + 1))
            elif r: st.add(int(r))
        out[j] = st
    return out
SKIP = parse_skip(a.skip); NB = len(tm.blocks); log("nodes", NODES, "sigmas", [round(s, 3) for s in SIG], "use", USE, "skip", {j: "%d blocks" % len(s) for j, s in SKIP.items()}, "src", a.src, "mode", a.mode)
# ---- block wrapper ----
CTX = dict(step=0, active=False); RES = {}   # RES[(step, L)] = residual of the last full run of block L at that step (bf16)
def kv_only(blk, L, x, e, grid_sizes, freqs, kv_cache, current_start, frame_seqlen):
    """write block L's K/V for this chunk from its input x (what the full block would write), without attention / FFN"""
    sa = blk.self_attn
    with torch.amp.autocast('cuda', dtype=torch.float32): em = (blk.modulation.unsqueeze(0) + e).chunk(6, dim=2)
    xm = (blk.norm1(x).float() * (1 + em[1].squeeze(2)) + em[0].squeeze(2)).to(x.dtype)
    b, s = xm.shape[:2]; n, d = sa.num_heads, sa.head_dim
    k = sa.norm_k(sa.k(xm)).view(b, s, n, d); v = sa.v(xm).view(b, s, n, d)
    csf = current_start // frame_seqlen; rk = M.causal_rope_apply(k, grid_sizes, freqs, start_frame=csf).type_as(v)
    size = kv_cache["k"].shape[1]; sink = sa.sink_size * frame_seqlen; c = current_start // s; sink_c = sink // s; slots = (size - sink) // s
    off = c * s if c < sink_c + slots else sink + ((c - sink_c) % slots) * s
    kv_cache["k"][:, off:off + s] = rk.detach(); kv_cache["v"][:, off:off + s] = v.detach()
def wrap(L, blk):
    full = torch.compile(blk.forward, dynamic=False) if a.compile else blk.forward
    def fwd(x, **kw):
        j = CTX["step"]
        if CTX["active"] and L in SKIP.get(j, ()):
            key = (j, L) if a.src == "chunk" else (j - 1, L)
            if key in RES:
                if a.kv_only: kv_only(blk, L, x, kw["e"], kw["grid_sizes"], kw["freqs"], kw["kv_cache"], kw["current_start"], kw["frame_seqlen"])
                out = x + RES[key].to(x.dtype)
                if a.src == "step": RES[(j, L)] = RES[key]
                return out
        out = full(x, **kw); RES[(j, L)] = (out - x).detach().to(torch.bfloat16); return out
    blk.forward = fwd
for L, blk in enumerate(tm.blocks): wrap(L, blk)
def gen(seed, c, k=0): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k * 100003)
@torch.no_grad()
def denoise(c, g, active):
    x = None; CTX["active"] = active
    for j, i in enumerate(USE):
        z = Tg.chunk_noise(c, g) if j == 0 else (1 - SIG[i]) * x + SIG[i] * Tg.randn_like(x, g)
        CTX["step"] = j; x = Tg.target_x0(z, i, c).float()
    CTX["active"] = False; return x
@torch.no_grad()
def commit(x, c):
    fs = Tg._frame_seqlen; L = Tg._chunk_size * fs; FP.ACTIVE["a"] = dict(idx=torch.arange(L - fs, L, device=dev), n=fs, L=L); CTX["step"] = -1
    try: Tg.commit(x, c)
    finally: FP.ACTIVE["a"] = None
def tokgap(xa, xb):   # per-token (2x2 latent patch per frame) relative L2 gap
    C, F_, H, W = xa.shape; h, w = H // 2, W // 2
    ta = xa[:, :, :h * 2, :w * 2].reshape(C, F_, h, 2, w, 2).permute(1, 2, 4, 0, 3, 5).reshape(F_ * h * w, -1); tb = xb[:, :, :h * 2, :w * 2].reshape(C, F_, h, 2, w, 2).permute(1, 2, 4, 0, 3, 5).reshape(F_ * h * w, -1)
    return (ta - tb).norm(dim=-1) / tb.norm(dim=-1).clamp_min(1e-6)
def prepare(task):
    T = ALL[task]; cond = Tg.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=a.frames, max_area=a.area); Tg.reset_caches(); RES.clear(); return cond["num_chunks"]
TASKS = a.tasks.split(","); SUMM = dict(args=vars(a), clips={})
N = prepare(TASKS[0]); g = gen(a.seed + 1, 0)
for c in range(min(N, a.warm + 2)): x = denoise(c, g, c >= a.warm); commit(x, c)   # warm-up / compile both paths
for task in TASKS:
    N = prepare(task); g = gen(a.seed, 0); chunks = []; rec = []; t0 = now()
    for c in range(N):
        active = c >= a.warm
        if a.mode == "tf":
            g_skip = gen(a.seed, c, 1); g_full = gen(a.seed, c, 1)   # same noise for both passes
            sync(); t1 = now(); xs = denoise(c, g_skip, active); sync(); ts = now() - t1
            t1 = now(); xf = denoise(c, g_full, False); sync(); tf_ = now() - t1; commit(xf, c); chunks.append(xf)
            gp = tokgap(xs, xf); rec.append(dict(c=c, active=active, skip_s=round(ts, 4), full_s=round(tf_, 4), rel=round(float((xs - xf).norm() / xf.norm()), 4), gap_med=round(float(gp.median()), 4), gap_q90=round(float(gp.quantile(0.9)), 4), within=[round(float((gp < t).float().mean()), 4) for t in (0.1, 0.2, 0.3, 0.5)]))
        else:
            sync(); t1 = now(); x = denoise(c, g, active); commit(x, c); sync(); rec.append(dict(c=c, active=active, chunk_s=round(now() - t1, 4))); chunks.append(x)
    od = a.out + "/" + task; os.makedirs(od, exist_ok=True); lat = torch.cat(chunks, 1); torch.save(lat.half().cpu(), od + "/lat.pt")
    if a.mode == "free" and a.video: Tg.decode_and_save([lat.to(dev)], od)
    act = [r for r in rec if r["active"]]
    if a.mode == "tf":
        summ = dict(task=task, n_active=len(act), skip_s=sorted(r["skip_s"] for r in act)[len(act) // 2], full_s=sorted(r["full_s"] for r in act)[len(act) // 2], rel=sum(r["rel"] for r in act) / len(act), gap_med=sum(r["gap_med"] for r in act) / len(act), within=[sum(r["within"][k] for r in act) / len(act) for k in range(4)])
        log("clip %s: skip %.3f s vs full %.3f s per chunk, rel gap %.3f, token gap med %.3f, within 0.1/0.2/0.3/0.5: %s" % (task, summ["skip_s"], summ["full_s"], summ["rel"], summ["gap_med"], " / ".join("%.0f %%" % (100 * v) for v in summ["within"])))
    else:
        summ = dict(task=task, chunk_s=sorted(r["chunk_s"] for r in act)[len(act) // 2], fps=round(12.0 / sorted(r["chunk_s"] for r in act)[len(act) // 2], 2), clip_s=round(now() - t0, 1), peak_GiB=round(torch.cuda.max_memory_allocated() / GiB, 2))
        log("clip %s: %.3f s/chunk = %.1f fps" % (task, summ["chunk_s"], summ["fps"]))
    json.dump(dict(summary=summ, chunks=rec), open(od + "/timing.json", "w"), indent=1); SUMM["clips"][task] = summ; json.dump(SUMM, open(a.out + "/summary.json", "w"), indent=1)
log("DONE", a.out)
