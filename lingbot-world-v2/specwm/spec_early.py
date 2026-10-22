"""Token-wise early exit for the 14B causal-fast (Speculative Forcing with our learned head): 4-step schedule; after step k < 4 the head
(scripts/head_train.py on gap_dump features) predicts each token's gap to the 4-step output from the last block's hidden state; tokens whose smoothed
prediction is <= eps are committed (x0 frozen, K/V kept from this step) and only the remaining tokens are forwarded in later steps (fast_patch.ACTIVE).
The chunk's clean pass (commit) then refreshes all K/V as usual. Modes: early (head), uniform:K (K steps for every token, same code path, for timing).
Reports mean denoising steps per token, gen time, and writes video.mp4 + loop.json.  1 GPU.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.spec_early --tasks a,b --seeds 1,2 --mode early --eps 0.24 --out DIR [--head H --warm 1]"""
import argparse, os, sys, json, time, math, numpy as np, torch, torch.nn.functional as F, imageio
if os.environ.get("RESERVE_GB"): import torch as _t; _r = _t.empty(int(os.environ["RESERVE_GB"]) << 30, dtype=_t.uint8, device="cuda:0"); del _r   # grab the GPU before other users' jobs do
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--tasks", required=True); ap.add_argument("--seeds", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--mode", default="early", help="early | uniform:K"); ap.add_argument("--eps", type=float, default=0.24); ap.add_argument("--arm", default="")
ap.add_argument("--head", default=S + "/lingbot-world-v2/specwm_out/head14b_v3/head.pt"); ap.add_argument("--data", default=S + "/lingbot-world-v2/specwm_out/gap_dump_v3")
ap.add_argument("--bucket", type=int, default=512); ap.add_argument("--warm", type=int, default=1); ap.add_argument("--no-smooth", action="store_true"); ap.add_argument("--no-video", action="store_true"); ap.add_argument("--frames", type=int, default=161); ap.add_argument("--stevo-out", default="", help="StEvo-Bench mode: all camera tasks (or --tasks), seed --seeds[0], video saved as DIR/<task>.mp4"); ap.add_argument("--shard", default="0/1"); ap.add_argument("--min-exit", type=int, default=1); ap.add_argument("--refresh-last", action="store_true"); ap.add_argument("--chunk-frac", type=float, default=0.0); ap.add_argument("--model", default="14b"); ap.add_argument("--t5-cpu", action="store_true", help="[10/17] keep the text encoder on the CPU (prepare slower, 11 GiB less)"); ap.add_argument("--window", type=int, default=18, help="[10/17 baselines] ring KV window in latent frames (sink 6 included); 12 at 1280x704"); ap.add_argument("--stream-decode", type=int, default=0, help="1: decode every committed chunk right away (side CUDA stream, separate VAE instance) and record per-frame ready times (wm_latency.py format)"); ap.add_argument("--commit", default="separate", choices=["separate", "partial1", "safe"], help="diag modes: separate = clean t=0 pass; partial1 = keep the K/V of the chunk's last denoising forward (fused) and re-commit only the last latent frame clean; safe = partial1 if the chunk stopped at step >= 3, else a full clean commit"); ap.add_argument("--no-ref", action="store_true", help="diag modes: skip the dense 4-step reference pass (only used for the err diagnostic) so wall time = method time"); ap.add_argument("--skip-commit", action="store_true", help="no clean pass: keep the K/V written during denoising (committed tokens: at their commit step)"); ap.add_argument("--chunk-size", type=int, default=3, help="latent frames per chunk (10/13 §23: 1 / 2 work on the stock model)"); ap.add_argument("--area", type=int, default=480 * 832, help="max_area for prepare (704x384 = 270336)"); ap.add_argument("--action-dir", default="", help="override every task's action path (poses.npy / intrinsics.npy dir), e.g. a data/actstream/<name> stream")
ap.add_argument("--long-clip", type=int, default=0, help="[10/19 EXP 1, default off = unchanged] 1 = long clips (P6: 969 poses): KV caches dropped before prepare, the --stream-decode side decoder keeps no decoded frames on the GPU (finish() result is unused), and video.mp4 is decoded latent frame by latent frame to the host (binding_v2.decode_frames_stream with this script's own CPU conversion: identical frames)")
ap.add_argument("--warm-frames", type=int, default=0, help="[10/19, default 0 = --frames] frames of the --warm clip (dynamic-shape compile needs ~14 chunks, not a whole 60-s clip)"); a = ap.parse_args()
if a.skip_commit: os.environ["SPECWM_SKIP_COMMIT"] = "1"
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
sys.path.insert(0, S + "/scripts")
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
if os.environ.get("SPECWM_TASKS"): ALL.update({t["id"]: t for t in json.load(open(os.environ["SPECWM_TASKS"]))})   # [10/16 baselines] extra task entries (same schema)
if a.model == "14b": B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=a.window, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
else: B = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=a.window, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), cfg_name="i2v-1.3B", assets_dir=FAST)
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True)
if a.model == "14b": FP.fp8_swap(B.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 64; _dyn.config.accumulated_recompile_limit = 4096
DYN = os.environ.get("SPEC_DYNAMIC", "1") == "1"   # dynamic shapes: one graph for every active-token count (static graphs recompile per bucket and hit the limit)
for blk in B.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=DYN)
if not a.t5_cpu: B.pipe.text_encoder.model.to("cuda:0")   # [10/17 baselines] --t5-cpu keeps umT5 on the CPU (binding default; saves 11 GiB at 1280x704)
_te = B.pipe.text_encoder
class _TE:
    def __init__(s, te): s.te = te; s.model = te.model
    def __call__(s, texts, device): return s.te(texts, torch.device("cpu" if a.t5_cpu else "cuda:0"))
B.pipe.text_encoder = _TE(_te); dev = "cuda:0"; sync = torch.cuda.synchronize
EARLY = a.mode == "early"; K_UNI = int(a.mode.split(":")[1]) if a.mode.startswith("uniform") else 4
DIAG = a.mode.split(":")[1] if a.mode.startswith("diag:") else None   # oracle_dense | oracle_sparse | learned_dense | learned_sparse (feedback 10/7, 5.1)
# ---- head + feature normalization (same as gap_dump.py)
class Head(torch.nn.Module):
    def __init__(s, cin=1536):
        super().__init__(); s.ln = torch.nn.LayerNorm(cin); s.c1 = torch.nn.Conv3d(cin, 96, 3, padding=1); s.c2 = torch.nn.Conv3d(96, 48, 3, padding=1); s.c3 = torch.nn.Conv3d(48, 1, 1)
    def forward(s, h): x = s.ln(h.float()).permute(0, 4, 1, 2, 3); x = F.gelu(s.c1(x)); x = F.gelu(s.c2(x)); return s.c3(x).squeeze(1)
HEAD = None
if EARLY:
    HEAD = Head().to(dev).eval(); HEAD.load_state_dict(torch.load(a.head, map_location=dev)["state"])
    PROJ = torch.load(a.data + "/proj.pt").to(dev, torch.bfloat16); NRM = torch.load(S + "/lingbot-world-v2/specwm_out/chan_norm_14b.pt"); MU, SD = NRM["mu"].to(dev), NRM["sd"].to(dev)
HID = {}
def hook(m, inp, out):
    h = out[0] if isinstance(out, tuple) else out; act = FP.ACTIVE["a"]; f = (((h[0].float() - MU) / SD).to(torch.bfloat16) @ PROJ)
    if act is None: HID["h"] = f
    else: HID["h"][act["idx"][:act["n"]]] = f[:act["n"]]
if EARLY: B.target_model.blocks[-1].register_forward_hook(hook)
EE = None
if DIAG: from specwm.early_exit import EarlyExit; EE = EarlyExit(B, eps=a.eps, bucket=a.bucket, smooth=not a.no_smooth, min_exit=a.min_exit, refresh_last=a.refresh_last, chunk_frac=a.chunk_frac, head=(a.head if a.model == "14b" else ""))
def gauss1d(n, sd): x = torch.arange(n, dtype=torch.float32) - n // 2; g = torch.exp(-x ** 2 / (2 * sd ** 2)); return g / g.sum()
G9 = (gauss1d(9, 1.0)[:, None] * gauss1d(9, 1.0)[None, :]).to(dev).view(1, 1, 9, 9)
def smooth(p):   # p [F, h, w] -> 9x9 spatial gaussian (reflect) then temporal gaussian (sd 1 frame) within the chunk
    f_ = p.shape[0]; q = F.conv2d(F.pad(p.unsqueeze(1), (4, 4, 4, 4), mode="reflect"), G9).squeeze(1)
    wt = torch.stack([gauss1d(2 * f_ - 1, 1.0)[f_ - 1 - i:2 * f_ - 1 - i] for i in range(f_)]).to(dev); wt = wt / wt.sum(1, keepdim=True)
    return torch.einsum("ij,jhw->ihw", wt, q)
def patch(x):   # [16, F, H, W] -> [F*(H/2)*(W/2), 64]
    c, f, h, w = x.shape; return x.view(c, f, 1, h // 2, 2, w // 2, 2).permute(1, 3, 5, 2, 4, 6, 0).reshape(f * (h // 2) * (w // 2), 64)
def unpatch(t, shape):
    c, f, h, w = shape; return t.view(f, h // 2, w // 2, 1, 2, 2, c).permute(6, 0, 3, 1, 4, 2, 5).reshape(c, f, h, w)
import threading
from torch.cuda import amp as _amp
class StreamDecoder:   # as in gen_fast.py / res_split_infer.py
    def __init__(s, vae, keep=True): s.v = vae; s.m = vae.model; s.m.clear_cache(); s.st = torch.cuda.Stream(); s.out = []; s.ready = []; s.nf = 0; s.q = []; s.lock = threading.Lock(); s.th = []; s.keep = keep   # [10/19] keep=False: no decoded frames retained on the GPU
    def _watch(s, ev, ids, chunk):
        ev.synchronize(); t = time.time()
        with s.lock: s.ready += [(i, chunk, t) for i in ids]
    def push(s, z, chunk=-1):
        ev = torch.cuda.Event(); ev.record()
        with torch.cuda.stream(s.st):
            s.st.wait_event(ev); z.record_stream(s.st)
            with torch.no_grad(), _amp.autocast(dtype=s.v.dtype):
                sc = s.v.scale; zz = z.unsqueeze(0) / sc[1].view(1, -1, 1, 1, 1) + sc[0].view(1, -1, 1, 1, 1); x = s.m.conv2(zz)
                for i in range(x.shape[2]):
                    s.m._conv_idx = [0]; o = s.m.decoder(x[:, :, i:i + 1], feat_cache=s.m._feat_map, feat_idx=s.m._conv_idx).float().clamp_(-1, 1)
                    if s.keep: s.out.append(o)
                    host = ((o[0] + 1) * 127.5).to(torch.uint8).to("cpu", non_blocking=True); e2 = torch.cuda.Event(); e2.record(s.st)
                    ids = list(range(s.nf, s.nf + o.shape[2])); s.nf += o.shape[2]; s.q.append(host)
                    th = threading.Thread(target=s._watch, args=(e2, ids, chunk)); th.start(); s.th.append(th)
    def finish(s):
        torch.cuda.current_stream().wait_stream(s.st); [t.join() for t in s.th]; im = torch.cat(s.out, 2)[0] if s.out else None; s.m.clear_cache(); return im
VAE2 = None
if a.stream_decode:
    from wan.modules.vae2_1 import Wan2_1_VAE
    VAE2 = Wan2_1_VAE(vae_pth=FAST + "/Wan2.1_VAE.pth", device="cuda:0", dtype=B.pipe.vae.dtype)
def gen(seed, c): return torch.Generator().manual_seed(seed * 1009 + c * 7)
from specwm.early_exit import patch as EE_patch
@torch.no_grad()
def run(task, seed, out, timed=True, frames=None):
    T = ALL[task]
    if a.long_clip: import gc as _gcm; B._caches = {}; _gcm.collect(); torch.cuda.empty_cache()   # [10/19] free the previous clip's KV before the long VAE encode (rebuilt below)
    sync(); t0 = time.time()
    cond = B.prepare(T["image"], T.get("camera_WM", ""), a.action_dir or T["action"], frame_num=frames or a.frames, max_area=a.area, chunk_size=a.chunk_size); B.reset_caches(); N = cond["num_chunks"]; sig = [float(s) for s in B.node_sigmas()]
    sync(); prep = time.time() - t0; SDc = (StreamDecoder(VAE2, keep=not a.long_clip) if a.long_clip else StreamDecoder(VAE2)) if VAE2 is not None else None; t0 = time.time(); chunks = []; depths = []; tm = dict(denoise=0.0, commit=0.0, verify=0.0); chunk_start = []; per_chunk = []
    for c in range(N):
        chunk_start.append(time.time()); g = gen(seed, c); z = B.chunk_noise(c, g); shape = tuple(z.shape); f_, h2, w2 = shape[1], shape[2] // 2, shape[3] // 2; L = f_ * h2 * w2
        if DIAG:
            xs = None; masks = None
            if not a.no_ref:
                t1 = time.time(); st_g = g.get_state(); xs = EE.dense_trajectory(c, z, g); masks = EE.oracle_masks(xs) if DIAG.startswith("oracle") else None; g.set_state(st_g); sync(); tm["verify"] += time.time() - t1   # reference pass (not part of the timed method)
            t1 = time.time(); x0_final, ms, hist = EE.gen_chunk(c, z, g, masks=masks, sparse=DIAG.endswith("sparse")); sync(); tm["denoise"] += time.time() - t1
            if xs is not None: tK = EE_patch(xs[-1]); err = float((((EE_patch(x0_final) - tK).norm(dim=1)) / float(tK.norm(dim=1).median())).mean())
            else: err = float("nan")
            stop = 1 + max(range(len(hist)), key=lambda k: hist[k])   # chunk mode: every token stops at the same step
            t1 = time.time()
            if a.commit == "separate" or (a.commit == "safe" and stop <= 2): B.commit(x0_final, c)
            else:   # fused: frames 0..F-2 keep the K/V of the last denoising forward; the newest latent frame is re-committed clean
                fs = B._frame_seqlen; FP.ACTIVE["a"] = dict(idx=torch.arange(L - fs, L, device=dev), n=fs, L=L)
                try: B.commit(x0_final, c)
                finally: FP.ACTIVE["a"] = None
            sync(); tm["commit"] += time.time() - t1
            if SDc is not None: SDc.push(x0_final.float(), c)
            chunks.append(x0_final); depths.append(ms); per_chunk.append(hist + [err]); continue
        x0_tok = torch.zeros(L, 64, device=dev); committed = torch.zeros(L, dtype=torch.bool, device=dev); commit_step = torch.full((L,), K_UNI, device=dev); FP.ACTIVE["a"] = None
        t1 = time.time()
        for i in range(K_UNI):
            active_idx = None if i == 0 else (~committed).nonzero().squeeze(1)
            if active_idx is not None:
                n = int(active_idx.numel());
                if n == 0: break
                pad = (-n) % a.bucket; idx = torch.cat([active_idx, active_idx[-1:].expand(pad)]) if pad else active_idx
                FP.ACTIVE["a"] = dict(idx=idx, n=n, L=L)
            x0 = B.target_x0(z, i, c).float(); xt = patch(x0)
            if active_idx is None: x0_tok = xt
            else: x0_tok[active_idx] = xt[active_idx]
            if EARLY and i < K_UNI - 1:
                sync(); tv = time.time()
                hp = F.avg_pool2d(HID["h"].view(f_, h2, w2, -1).permute(0, 3, 1, 2).float(), 2, ceil_mode=True).permute(0, 2, 3, 1)   # [F, h4, w4, 1536]
                pred = HEAD(hp.unsqueeze(0))[0]; pred = F.interpolate(pred.unsqueeze(0), size=(h2, w2), mode="nearest")[0]
                if not a.no_smooth: pred = smooth(pred)
                acc = (pred.flatten() <= a.eps) & ~committed; commit_step[acc] = i + 1; committed |= acc
                tm["verify"] += time.time() - tv
            if i < K_UNI - 1:
                x0_full = unpatch(x0_tok, shape); s_ = sig[i + 1]; z = (1 - s_) * x0_full + s_ * B.randn_like(x0_full, g)
        FP.ACTIVE["a"] = None; x0_final = unpatch(x0_tok, shape); sync(); tm["denoise"] += time.time() - t1
        t1 = time.time(); B.commit(x0_final, c); sync(); tm["commit"] += time.time() - t1
        if SDc is not None: SDc.push(x0_final.float(), c)   # 10/13: uniform path never fed the stream decoder (finish() crashed on an empty list)
        chunks.append(x0_final); depths.append(float(commit_step.float().mean())); per_chunk.append([int((commit_step == k).sum()) for k in range(1, 5)])
    sync(); gen_s = time.time() - t0; dec_s = None; fr_ready = None
    if SDc is not None: t1 = time.time(); SDc.finish(); sync(); dec_s = time.time() - t1; fr_ready = sorted(SDc.ready)
    if not a.no_video:
        t1 = time.time()
        if a.long_clip:   # [10/19] same calls and the same CPU conversion per decoded latent frame, frames accumulated on the host (no ~31 GB GPU clip at 1280x704)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16): vu8 = B.decode_frames_stream(torch.cat(chunks, 1), post=lambda o: (((o.float().clamp(-1, 1).cpu() + 1) / 2).permute(1, 2, 3, 0).numpy() * 255).astype("uint8"))
            sync(); dec_s = time.time() - t1; os.makedirs(out, exist_ok=True); imageio.mimwrite(out + "/video.mp4", list(vu8), fps=16, quality=8)
        else:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16): im = B.pipe.vae.decode([torch.cat(chunks, 1).to(dev)])[0].float().clamp(-1, 1).cpu()
            sync(); dec_s = time.time() - t1; os.makedirs(out, exist_ok=True)
            v = ((im + 1) / 2).permute(1, 2, 3, 0).numpy(); imageio.mimwrite(out + "/video.mp4", list((v * 255).astype("uint8")), fps=16, quality=8)
        if a.stevo_out: os.makedirs(a.stevo_out, exist_ok=True); imageio.mimwrite("%s/%s.mp4" % (a.stevo_out, task), list((v * 255).astype("uint8")), fps=16, quality=8)
    os.makedirs(out, exist_ok=True)
    if os.environ.get("SAVE_LAT", "1") != "0": torch.save(torch.cat(chunks, 1).half().cpu(), out + "/lat.pt")
    R = dict(task=task, seed=seed, mode=a.mode, eps=a.eps, chunks=N, mean_steps=float(np.mean(depths)), per_chunk_commit_hist=per_chunk, prepare_s=round(prep, 2), gen_s=round(gen_s, 2),
             decode_s=None if dec_s is None else round(dec_s, 2), e2e_s=round(prep + gen_s + (dec_s or 0), 2), times={k: round(v, 2) for k, v in tm.items()}, chunk_start=chunk_start, t_gen_start=t0, frame_ready=fr_ready, stream_decode=a.stream_decode)
    json.dump(R, open(out + "/loop.json", "w"), indent=1)
    print("[EARLY_DONE] %s s%d %s eps %g mean_steps %.2f gen %.1fs (denoise %.1f verify %.1f commit %.1f) prep %.1f dec %s e2e %.1f" % (task, seed, a.mode, a.eps, R["mean_steps"], gen_s, tm["denoise"], tm["verify"], tm["commit"], prep, dec_s, R["e2e_s"]), flush=True)
arm = a.arm or (("cs%d_" % a.chunk_size if a.chunk_size != 3 else "") + ("a%d_" % a.area if a.area != 480 * 832 else "") + (("early%g" % a.eps) if EARLY else ("diag_%s_%g" % (DIAG, a.eps) if DIAG else "uniform%d" % K_UNI)) + ("_skipcommit" if a.skip_commit else "") + ("_min%d" % a.min_exit if a.min_exit > 1 else "") + ("_refresh" if a.refresh_last else "") + ("_chunk%g" % a.chunk_frac if a.chunk_frac > 0 else "") + ("_1p3b" if a.model != "14b" else "") + ("_" + os.path.basename(a.action_dir) if a.action_dir else ""))
cases = [(t, int(s)) for t in a.tasks.split(",") for s in a.seeds.split(",")]
if a.stevo_out:
    si, sn = (int(x) for x in a.shard.split("/")); tl = sorted(ALL) if a.tasks == "all" else a.tasks.split(","); cases = [(t, int(a.seeds.split(",")[0])) for t in tl[si::sn] if not os.path.exists("%s/%s.mp4" % (a.stevo_out, t))]
if a.warm: run(cases[0][0], cases[0][1], "%s/%s/%s_warm" % (a.out, cases[0][0], arm), frames=a.warm_frames or None)
for t, s in cases:
    if os.path.exists("%s/%s/%s_s%d/loop.json" % (a.out, t, arm, s)): print("[skip]", t, s, flush=True); continue
    run(t, s, "%s/%s/%s_s%d" % (a.out, t, arm, s))
print("[ALL_DONE]", flush=True)
