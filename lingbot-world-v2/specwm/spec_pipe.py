"""Two-GPU pipelined speculative AR loop. rank 0 = target (14B, its own GPU), rank 1 = draft (1.3B, its own GPU); latents travel over gloo (CPU).
Per chunk c (chunk 0 = target):
  target: verify = 14B node-0 x0 from the chunk's noise (independent of the draft) | receive draft node-0 x0 -> d = ||xT - xD||^2/||xT||^2
          accept (d <= tau): receive the draft's finished chunk, commit it | reject: finish with the 14B's remaining nodes, commit, send the chunk to the draft
  draft : node 0 -> send; remaining nodes -> send; commit own chunk; then (lookahead 1) draft c+1 while the target verifies/commits c.
          On a reject of c it overwrites its KV slot with the target's chunk and redrafts c+1. It also VAE-decodes every final chunk on a side stream.
Noise is derived per chunk from (seed, c, path) so both processes draw identical node-0 noise.
modes: --tau -1 target-only (draft process only decodes), --tau 1e9 accept all; --verifier critic:PATH:TAU uses a latent critic on the draft node-0 x0
instead of the 14B verify forward (the 14B then only commits / finishes rejects).
usage: two processes,  CUDA_VISIBLE_DEVICES=gT python -m specwm.spec_pipe --rank 0 ... & CUDA_VISIBLE_DEVICES=gD python -m specwm.spec_pipe --rank 1 ...  (same args)"""
import argparse, os, sys, json, time, collections, numpy as np, torch, imageio, threading
import torch.distributed as dist
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--rank", type=int, required=True); ap.add_argument("--port", type=int, default=29611)
ap.add_argument("--tasks", default="easy_urban_rain"); ap.add_argument("--seeds", default="1234"); ap.add_argument("--out", required=True); ap.add_argument("--arm", default="")
ap.add_argument("--tau", type=float, default=0.21); ap.add_argument("--verifier", default="agree", help="agree | critic:PATH:TAU (accept if critic logit >= TAU)")
ap.add_argument("--t-steps", default="0,500"); ap.add_argument("--d-steps", default="0,250,500,750"); ap.add_argument("--d-fp8", type=int, default=1)
ap.add_argument("--d-lora", default="", help="distilled draft LoRA (draft_distill_train.py), merged into the 1.3B weights"); ap.add_argument("--no-video", action="store_true"); ap.add_argument("--warm", type=int, default=1, help="run the first case once untimed (compile warm-up)"); ap.add_argument("--force-first", type=int, default=1, help="leading chunks always made by the target (SDVG: 1)"); ap.add_argument("--t-early", type=float, default=0.0, help="target = token-wise early exit on the 4-step schedule with this eps (overrides --t-steps)"); ap.add_argument("--anchor", type=float, default=0.0, help="anchor split: the target runs step 1 (4-step schedule) and sends x0 to the draft, which finishes with its last --anchor-dsteps steps; accept if the token gap to the anchor <= this value (sigma units; 1e9 = always)"); ap.add_argument("--anchor-dsteps", type=int, default=2); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
dist.init_process_group("gloo", init_method="tcp://127.0.0.1:%d" % a.port, rank=a.rank, world_size=2)
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
TARGET_ONLY = a.tau < 0 and a.verifier == "agree"; CRIT = a.verifier.startswith("critic:"); ANCHOR = a.anchor > 0
TI = (0, 250, 500, 750) if (a.t_early > 0 or a.anchor > 0) else tuple(int(x) for x in a.t_steps.split(",")); DI = tuple(int(x) for x in a.d_steps.split(","))
if a.rank == 0: B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=TI)
else: B = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=DI, cfg_name="i2v-1.3B", assets_dir=FAST)
if a.rank == 1 and a.d_lora:   # merge W += B A in block order q,k,v,o[,ffn0,ffn2]
    st = torch.load(a.d_lora, map_location="cpu"); it = iter(zip(st["A"], st["B"]))
    with torch.no_grad():
        for blk in B.target_model.blocks:
            lins = [getattr(blk.self_attn, n) for n in ("q", "k", "v", "o")] + ([blk.ffn[0], blk.ffn[2]] if st["ffn"] else [])
            for lin in lins: A_, B_ = next(it); lin.weight += (B_.float() @ A_.float()).to(lin.weight.device, lin.weight.dtype)
    print("[spec_pipe] merged draft LoRA", a.d_lora, "step", st.get("step"), flush=True)
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True)
if a.rank == 0 or a.d_fp8: FP.fp8_swap(B.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 64; _dyn.config.accumulated_recompile_limit = 4096
for blk in B.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=(a.rank == 0 and a.t_early > 0))
EE = None
if a.rank == 0 and a.t_early > 0: from specwm.early_exit import EarlyExit; EE = EarlyExit(B, eps=a.t_early)
B.pipe.text_encoder.model.to("cuda:0"); _te = B.pipe.text_encoder
class _TE:
    def __init__(s, te): s.te = te; s.model = te.model
    def __call__(s, texts, device): return s.te(texts, torch.device("cuda:0"))
B.pipe.text_encoder = _TE(_te)
_enc = B.pipe.vae.encode; _TAIL = {}
def _enc_head(videos, K=14):   # exact head (first K latent frames) + cached image-independent tail
    x = videos[0]; nf = 1 + 4 * (K - 1)
    if len(videos) != 1 or x.shape[1] <= nf or float(x[:, 1:].abs().max()) != 0.0: return _enc(videos)
    key = tuple(x.shape)
    if key not in _TAIL: y = _enc(videos)[0]; _TAIL[key] = y[:, K:].clone(); return [y]
    return [torch.cat([_enc([x[:, :nf]])[0], _TAIL[key]], 1)]
B.pipe.vae.encode = _enc_head
CS = None
if CRIT and a.rank == 1:
    from specwm.critic_model import CriticScorer
    _p, _, _t = a.verifier[len("critic:"):].rpartition(":"); CS = (CriticScorer(_p, device="cuda:0"), float(_t))
sync = torch.cuda.synchronize
SHAPE = None
def patch_tok(x):
    c_, f, h, w = x.shape; return x.view(c_, f, 1, h // 2, 2, w // 2, 2).permute(1, 3, 5, 2, 4, 6, 0).reshape(f * (h // 2) * (w // 2), 64)
def noise(seed, c, path, like=None):
    g = torch.Generator().manual_seed(seed * 1009 + c * 7 + {"node0": 0, "D": 1, "T": 2}[path]); return g
# ---- messaging: header [kind, c, flag, epoch] (int64) then an optional latent; non-blocking sends (buffers kept alive), FIFO per pair.
# epoch = number of rejects the draft has processed; the target drops draft messages from an older epoch (speculative work on a rejected branch).
K_XD0, K_XD, K_DEC, K_START, K_END, K_CRIT, K_ANCH = 1, 2, 3, 4, 5, 6, 7
PENDING = []
def send(kind, c, flag=0, x=None, epoch=0):
    h = torch.tensor([kind, c, flag, epoch], dtype=torch.int64); PENDING.append((dist.isend(h, dst=1 - a.rank), h))
    if x is not None: xb = x.detach().float().cpu().contiguous(); PENDING.append((dist.isend(xb, dst=1 - a.rank), xb))
    if len(PENDING) > 64: [w.wait() for w, _ in PENDING[:32]]; del PENDING[:32]
def flush(): [w.wait() for w, _ in PENDING]; PENDING.clear()
HAS_X = {K_XD0, K_XD, K_ANCH}   # kinds that carry a latent (K_DEC carries one only when flag == 0, i.e. reject)
def recv_hdr():
    h = torch.empty(4, dtype=torch.int64); dist.recv(h, src=1 - a.rank); return [int(v) for v in h]
def recv_x():
    x = torch.empty(SHAPE, dtype=torch.float32); dist.recv(x, src=1 - a.rank); return x
def recv_want(kind, c, epoch):
    """target side: read draft messages until (kind, c, epoch); drop stale ones (and their payloads)"""
    while True:
        h = recv_hdr(); has = h[0] in HAS_X
        if h[0] == kind and h[1] == c and h[3] == epoch: return h, (recv_x() if has else None)
        if has: recv_x()

class StreamDecoder:
    """bf16 VAE decode per final chunk on a side stream; per decoded latent frame: copy to host + event; a watcher thread logs when it completes"""
    def __init__(s, vae): s.v = vae; s.m = vae.model; s.m.clear_cache(); s.st = torch.cuda.Stream(); s.out = []; s.ready = []; s.nf = 0; s.q = []; s.lock = threading.Lock(); s.th = []
    def _watch(s, ev, ids, chunk):
        ev.synchronize(); t = time.time()
        with s.lock: s.ready += [(i, chunk, t) for i in ids]
    def push(s, z, chunk=-1):
        ev = torch.cuda.Event(); ev.record()
        with torch.cuda.stream(s.st):
            s.st.wait_event(ev); z.record_stream(s.st)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                sc = s.v.scale; zz = z.unsqueeze(0) / sc[1].view(1, -1, 1, 1, 1) + sc[0].view(1, -1, 1, 1, 1); x = s.m.conv2(zz)
                for i in range(x.shape[2]):
                    s.m._conv_idx = [0]; o = s.m.decoder(x[:, :, i:i + 1], feat_cache=s.m._feat_map, feat_idx=s.m._conv_idx).float().clamp_(-1, 1); s.out.append(o)
                    host = ((o[0] + 1) * 127.5).to(torch.uint8).to("cpu", non_blocking=True); e2 = torch.cuda.Event(); e2.record(s.st)
                    ids = list(range(s.nf, s.nf + o.shape[2])); s.nf += o.shape[2]; s.q.append(host)
                    th = threading.Thread(target=s._watch, args=(e2, ids, chunk)); th.start(); s.th.append(th)
    def finish(s):
        torch.cuda.current_stream().wait_stream(s.st); [t.join() for t in s.th]; im = torch.cat(s.out, 2)[0]; s.m.clear_cache(); return im

@torch.no_grad()
def target_case(task, seed, out):
    global SHAPE
    T = ALL[task]; sync(); t0 = time.time(); send(K_START, 0)
    cond = B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); B.reset_caches(); SHAPE = B._chunk_shape
    N = cond["num_chunks"]; sig = [float(s) for s in B.node_sigmas()]; recv_want(K_START, 0, 0); ep = 0   # draft ready (prepared)
    sync(); prep = time.time() - t0; t0 = time.time(); plan = ""; ds = []; tm = collections.defaultdict(float)
    def gen_target(c, x0):
        g = noise(seed, c, "T"); x = x0
        if EE is not None:   # early exit continues from the step-1 x0 (its hidden state is already in the buffer)
            x, ms, hist = EE.gen_chunk(c, B.chunk_noise(c, noise(seed, c, "node0")), g, x0_1=x0); steps.append(ms); return x
        for i in range(1, len(sig)):
            s_ = sig[i]; z = (1 - s_) * x + s_ * B.randn_like(x, g); x = B.target_x0(z, i, c)
        return x
    chunk_start = []; steps = []
    for c in range(N):
        chunk_start.append(time.time()); z = B.chunk_noise(c, noise(seed, c, "node0"))
        if c < a.force_first or TARGET_ONLY:
            t1 = time.time(); x = gen_target(c, B.target_x0(z, 0, c)).float(); sync(); tm["T_denoise"] += time.time() - t1; acc = False
        elif ANCHOR:   # cloud: step 1 (the layout) -> device finishes -> verify against the anchor (free) -> commit
            t1 = time.time(); xt = B.target_x0(z, 0, c).float(); sync(); tm["T_anchor"] += time.time() - t1; send(K_ANCH, c, 0, xt, epoch=ep)
            t1 = time.time(); h, xd = recv_want(K_XD, c, ep); xd = xd.to("cuda:0"); tm["wait_xd"] += time.time() - t1
            tt = patch_tok(xt); d = float(((patch_tok(xd) - tt).norm(dim=1) / float(tt.norm(dim=1).median())).mean()); acc = d <= a.anchor; ds.append(round(d, 4))
            if acc: x = xd
            else: t1 = time.time(); x = gen_target(c, xt).float(); sync(); tm["T_denoise"] += time.time() - t1
        elif CRIT:
            t1 = time.time(); h, _ = recv_want(K_CRIT, c, ep); acc = bool(h[2]); tm["wait_crit"] += time.time() - t1   # the draft judged its own node-0 x0
            if acc: t1 = time.time(); h, x = recv_want(K_XD, c, ep); x = x.to("cuda:0"); tm["wait_xd"] += time.time() - t1
            else: t1 = time.time(); x = gen_target(c, B.target_x0(z, 0, c)).float(); sync(); tm["T_denoise"] += time.time() - t1
        else:
            t1 = time.time(); xt = B.target_x0(z, 0, c); sync(); tm["T_verify"] += time.time() - t1
            t1 = time.time(); h, xd0 = recv_want(K_XD0, c, ep); xd0 = xd0.to("cuda:0"); tm["wait_xd0"] += time.time() - t1
            d = float(((xt - xd0) ** 2).mean() / (xt ** 2).mean().clamp_min(1e-8)); acc = d <= a.tau; ds.append(round(d, 4))
            if acc: t1 = time.time(); h, x = recv_want(K_XD, c, ep); x = x.to("cuda:0"); tm["wait_xd"] += time.time() - t1
            else: t1 = time.time(); x = gen_target(c, xt).float(); sync(); tm["T_denoise"] += time.time() - t1
        plan += "D" if acc else "T"
        t1 = time.time(); B.commit(x, c); sync(); tm["T_commit"] += time.time() - t1
        send(K_DEC, c, int(acc), None if acc else x)     # decision; the draft decodes the final chunk and (on reject) re-commits it
        if not acc and c >= a.force_first and not TARGET_ONLY: ep += 1
    sync(); gen = time.time() - t0; send(K_END, 0); recv_want(K_END, 0, ep); flush()   # draft finishes the decode tail
    tail = time.time() - t0 - gen
    R = dict(task=task, seed=seed, tau=a.tau, verifier=a.verifier, plan=plan, accepted=plan[1:].count("D"), chunks=N, prepare_s=round(prep, 2), gen_s=round(gen, 2), decode_tail_s=round(tail, 2),
             e2e_s=round(prep + gen + tail, 2), t_early=a.t_early, mean_steps=(sum(steps) / len(steps) if steps else None), chunk_start=chunk_start, t_gen_start=t0, times={k: round(v, 2) for k, v in tm.items()}, d=ds)
    os.makedirs(out, exist_ok=True); json.dump(R, open(out + "/loop.json", "w"), indent=1)
    print("[PIPE_DONE] %s s%d %s plan %s gen %.1fs e2e %.1fs | %s" % (task, seed, a.verifier if CRIT else "tau %g" % a.tau, plan, gen, R["e2e_s"], ", ".join("%s %.1f" % kv for kv in R["times"].items())), flush=True)

@torch.no_grad()
def draft_case(task, seed, out):
    global SHAPE
    T = ALL[task]; h = recv_hdr(); assert h[0] == K_START
    cond = B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); B.reset_caches(); SHAPE = B._chunk_shape
    N = cond["num_chunks"]; sig = [float(s) for s in B.node_sigmas()]; SD = None if a.no_video else StreamDecoder(B.pipe.vae); send(K_START, 0)
    crit_log = []; dstart = {}; final = {}; mine = {}; c_next = 1; tm = collections.defaultdict(float); decided = -1; ep = 0
    def handle(h):   # a decision from the target for chunk h[1]
        nonlocal decided, c_next, ep
        assert h[0] == K_DEC, h
        c, acc = h[1], bool(h[2]); x = mine.get(c) if acc else recv_x().to("cuda:0")
        if not acc:
            if not TARGET_ONLY: t1 = time.time(); B.commit(x, c); sync(); tm["D_commit_fix"] += time.time() - t1
            if c >= a.force_first and not TARGET_ONLY: ep += 1
            for k in [k for k in mine if k > c]: del mine[k]
            c_next = c + 1     # speculative work after c is void; also advances past the forced leading chunks
        final[c] = x; decided = c
        if SD is not None: SD.push(x, c)
    def draft(c):
        dstart.setdefault(c, time.time()); z = B.chunk_noise(c, noise(seed, c, "node0")); t1 = time.time(); x = B.target_x0(z, 0, c)
        if not TARGET_ONLY and not CRIT: send(K_XD0, c, 0, x, epoch=ep)
        g = noise(seed, c, "D")
        for i in range(1, len(sig)):
            s_ = sig[i]; zz = (1 - s_) * x + s_ * B.randn_like(x, g); x = B.target_x0(zz, i, c)
        x = x.float(); sync(); tm["D_denoise"] += time.time() - t1
        if CRIT:   # latent critic on the finished draft chunk (trained on finished chunks), context = previous final/own chunk
            t1 = time.time(); prev = mine.get(c - 1, final.get(c - 1)); l, _ = CS[0](x, prev, B._emb_chunks[c].squeeze(0)); acc = (l <= CS[1]) if CS[0].kind == "drift" else (l >= CS[1])
            crit_log.append((c, round(float(l), 3))); tm["D_critic"] += time.time() - t1; send(K_CRIT, c, int(acc), epoch=ep)
            if not acc: return None
        if not TARGET_ONLY: send(K_XD, c, 0, x, epoch=ep)
        t1 = time.time(); B.commit(x, c); sync(); tm["D_commit"] += time.time() - t1; return x
    while True:
        # lookahead 1: the draft may work on c_next only if the target has decided c_next - 2 (so a rollback needs at most one slot rewrite)
        blocked = CRIT and (c_next - 1) not in mine and decided < c_next - 1   # critic rejected the previous chunk: wait for the target's version
        if ANCHOR:   # no speculation: wait for the anchor of the next chunk, finish it with the last --anchor-dsteps steps, send it back
            h = recv_hdr()
            if h[0] == K_END: break
            if h[0] == K_ANCH:
                c = h[1]; x1 = recv_x().to("cuda:0"); dstart.setdefault(c, time.time()); t1 = time.time(); x = x1; g = noise(seed, c, "D")
                for i in range(len(sig) - a.anchor_dsteps, len(sig)): s_ = sig[i]; zz = (1 - s_) * x + s_ * B.randn_like(x, g); x = B.target_x0(zz, i, c)
                x = x.float(); sync(); tm["D_denoise"] += time.time() - t1; send(K_XD, c, 0, x, epoch=ep); mine[c] = x
                t1 = time.time(); B.commit(x, c); sync(); tm["D_commit"] += time.time() - t1; continue
            handle(h); continue
        if TARGET_ONLY or c_next >= N or c_next < a.force_first or decided < c_next - 2 or (c_next == a.force_first and decided < a.force_first - 1) or blocked:
            h = recv_hdr()
            if h[0] == K_END: break
            handle(h); continue
        # drain decisions that already arrived (non-blocking probe is not available in gloo; decisions are handled at the next wait)
        c = c_next; c_next += 1
        x = draft(c)
        if x is not None: mine[c] = x
    if SD is not None:
        im = SD.finish(); os.makedirs(out, exist_ok=True); v = ((im + 1) / 2).permute(1, 2, 3, 0).cpu().numpy()
        threading.Thread(target=lambda: imageio.mimwrite(out + "/video.mp4", list((v * 255).astype("uint8")), fps=16, quality=8)).start()
    send(K_END, 0, epoch=ep); flush(); os.makedirs(out, exist_ok=True)
    torch.save(torch.cat([final[c].cpu() for c in range(N)], 1).half(), out + "/lat.pt"); json.dump(dict(times={k: round(v, 2) for k, v in tm.items()}, critic=crit_log, chunk_start_draft=dstart, frame_ready=(sorted(SD.ready) if SD is not None else None)), open(out + "/draft_times.json", "w"))

arm = a.arm or ("target" if TARGET_ONLY else ("anchor%g_d%d" % (a.anchor, a.anchor_dsteps) if ANCHOR else ("crit" if CRIT else "spec_tau%g" % a.tau))) + ("" if a.force_first == 1 or TARGET_ONLY else "_ff%d" % a.force_first) + ("_early%g" % a.t_early if a.t_early > 0 else "")
cases = [(t, int(s)) for t in a.tasks.split(",") for s in a.seeds.split(",")]
if a.warm: cases = [cases[0] + ("warm",)] + cases
for cs in cases:
    task, seed = cs[0], cs[1]; out = "%s/%s/%s_%s" % (a.out, task, arm, "warm" if len(cs) > 2 else "s%d" % seed)
    (target_case if a.rank == 0 else draft_case)(task, seed, out)
dist.barrier(); print("[ALL_DONE]", flush=True) if a.rank == 0 else None
