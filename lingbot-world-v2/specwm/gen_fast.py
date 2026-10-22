"""Target-only (14B) or draft-only (1.3B) AR rollouts with the speed variants, many seeds/tasks, timed per component; output like spec_ar_loop:
ROOT/<task>/<arm>_s<seed>/{lat.pt, loop.json, video.mp4}. variants: --compile, --fp8 (W8A8 e4m3 linears in the blocks), fast_patch flags always on.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.gen_fast --model 14b --arm t14_fp8c --fp8 --compile --tasks a,b --seeds 1,2 --out ROOT"""
import argparse, os, sys, json, time, numpy as np, torch, imageio
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="14b"); ap.add_argument("--arm", required=True); ap.add_argument("--fp8", action="store_true"); ap.add_argument("--compile", action="store_true")
ap.add_argument("--tasks", default="sand_filling_up_jar,sandcastle_wave_impact"); ap.add_argument("--seeds", default="1234,4321,9999,101,102,103"); ap.add_argument("--out", required=True)
ap.add_argument("--max-area", type=int, default=480 * 832); ap.add_argument("--no-video", action="store_true"); ap.add_argument("--t5-gpu", action="store_true"); ap.add_argument("--stream-decode", action="store_true"); ap.add_argument("--check-decode", action="store_true"); ap.add_argument("--vae-head", type=int, default=0, help="encode only the first K latent frames of the (image + zero frames) conditioning video; the image-independent tail is cached per shape"); ap.add_argument("--vae-bf16-decode", action="store_true"); ap.add_argument("--orig", action="store_true", help="no fast_patch at all (original code path)"); ap.add_argument("--sage", action="store_true", help="SageAttention2 (INT8 QK, FP8 PV) for self-attention"); ap.add_argument("--lora", default="", help="draft LoRA (draft_distill_train.py) merged into the weights"); ap.add_argument("--steps", default="0,250,500,750", help="timestep indices of the sampling nodes"); ap.add_argument("--sp", action="store_true", help="Ulysses sequence parallel over the torchrun world (specwm/fast_sp.py)"); ap.add_argument("--decode-rank", action="store_true", help="with --sp: the last torchrun rank only VAE-decodes (latents sent by rank 0), the others form the SP group"); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
RANK0 = True; DEV = "cuda:0"
if a.sp:
    import torch.distributed as dist
    LR = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(LR); dist.init_process_group("nccl", device_id=torch.device("cuda", LR)); RANK0 = dist.get_rank() == 0; DEV = "cuda:%d" % LR
    WS = dist.get_world_size(); DEC = WS - 1 if a.decode_rank else None
    SPG = dist.new_group(list(range(WS - 1))) if a.decode_rank else None
    if a.decode_rank and dist.get_rank() == DEC:   # ---- decode worker: VAE only; receives [16, 3, H, W] chunks from rank 0, decodes as they arrive
        from wan.modules.vae2_1 import Wan2_1_VAE
        vae = Wan2_1_VAE(vae_pth=S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast/Wan2.1_VAE.pth", device=DEV)
        hdr = torch.zeros(6, dtype=torch.int64, device=DEV)
        while True:
            dist.recv(hdr, src=0); N_, C_, T_, H_, W_, case = [int(v) for v in hdr]
            if N_ < 0: break
            out = open(S + "/exp/queue/.declist", "r").read().split("\n")[case] if False else None
            buf = torch.empty((C_, T_, H_, W_), dtype=torch.float32, device=DEV); m = vae.model; m.clear_cache(); frames = []; ready = []; nf = 0
            for c_ in range(N_):
                dist.recv(buf, src=0)
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16 if a.vae_bf16_decode else torch.float32):
                    sc = vae.scale; zz = buf.unsqueeze(0) / sc[1].view(1, -1, 1, 1, 1) + sc[0].view(1, -1, 1, 1, 1); xx = m.conv2(zz)
                    for i in range(xx.shape[2]):
                        m._conv_idx = [0]; o = m.decoder(xx[:, :, i:i + 1], feat_cache=m._feat_map, feat_idx=m._conv_idx).float().clamp_(-1, 1)
                        host = ((o[0] + 1) * 127.5).to(torch.uint8).cpu(); t_ = time.time(); frames.append(host); ready += [(nf + j, c_, t_) for j in range(o.shape[2])]; nf += o.shape[2]
            m.clear_cache(); dpath = torch.empty(1, dtype=torch.int64, device=DEV)
            name = torch.empty(512, dtype=torch.uint8, device=DEV); dist.recv(name, src=0); outdir = bytes(name.cpu().tolist()).rstrip(b"\x00").decode()
            v = torch.cat(frames, 1).permute(1, 2, 3, 0).numpy(); imageio.mimwrite(outdir + "/video.mp4", list(v), fps=16, quality=8)
            json.dump(dict(frame_ready=ready, decode_gpu=DEV), open(outdir + "/decode.json", "w")); print("[DECODE_DONE]", outdir, flush=True)
        dist.barrier(); dist.destroy_process_group(); sys.exit(0)
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"; TI = tuple(int(x) for x in a.steps.split(","))
B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=TI, device=DEV) if a.model == "14b" else \
    V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=TI, cfg_name="i2v-1.3B", assets_dir=FAST, device=DEV)
if a.lora:   # merge W += B A in block order q,k,v,o[,ffn0,ffn2]
    st = torch.load(a.lora, map_location="cpu"); it = iter(zip(st["A"], st["B"]))
    with torch.no_grad():
        for blk in B.target_model.blocks:
            for lin in [getattr(blk.self_attn, n) for n in ("q", "k", "v", "o")] + ([blk.ffn[0], blk.ffn[2]] if st["ffn"] else []):
                A_, B_ = next(it); lin.weight += (B_.float() @ A_.float()).to(lin.weight.device, lin.weight.dtype)
    print("[gen_fast] merged LoRA", a.lora, "step", st.get("step"), flush=True)
if not a.orig: FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage" if a.sage else "flash")
if a.sp:
    from specwm import fast_sp as FSP
    if a.decode_rank: FSP.set_group(SPG)
    FSP.apply_sp()
if a.fp8: FP.fp8_swap(B.target_model, True, keep=False)
if a.compile:
    import torch._dynamo as _dyn; _dyn.config.recompile_limit = 64; _dyn.config.accumulated_recompile_limit = 4096   # the KV window grows chunk by chunk -> one graph per length (default limit 8 silently fell back to eager)
    for blk in B.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
sync = torch.cuda.synchronize
if a.t5_gpu:   # T5 umt5-xxl to the GPU (was CPU: ~6 s per prepare)
    B.pipe.text_encoder.model.to(DEV); _te = B.pipe.text_encoder
    class _TE:
        def __init__(s, te): s.te = te; s.model = te.model
        def __call__(s, texts, device): return s.te(texts, torch.device(DEV))
    B.pipe.text_encoder = _TE(_te)
import threading
if a.vae_head:
    _enc = B.pipe.vae.encode; _TAIL = {}
    def _enc_head(videos):
        x = videos[0]; K = a.vae_head; nf = 1 + 4 * (K - 1)
        if len(videos) != 1 or x.shape[1] <= nf or float(x[:, 1:].abs().max()) != 0.0: return _enc(videos)
        key = tuple(x.shape)
        if key not in _TAIL: y = _enc(videos)[0]; _TAIL[key] = y[:, K:].clone(); return [y]   # first call per shape: full encode, cache the tail
        return [torch.cat([_enc([x[:, :nf]])[0], _TAIL[key]], 1)]
    B.pipe.vae.encode = _enc_head
from torch.cuda import amp as _amp
class StreamDecoder:
    """Wan2.1 VAE decode one committed latent chunk at a time, on a side CUDA stream (same per-frame causal feature cache as vae.decode -> same frames).
    For latency: after every decoded latent frame the pixel frames are copied to pinned host memory and an event is recorded; a watcher thread
    stores the wall-clock time at which each event completes (ready[i] = (pixel_frame_id, chunk, time))."""
    def __init__(s, vae): s.v = vae; s.m = vae.model; s.m.clear_cache(); s.st = torch.cuda.Stream(); s.out = []; s.ready = []; s.nf = 0; s.q = []; s.lock = threading.Lock(); s.th = []
    def _watch(s, ev, ids, chunk):
        ev.synchronize(); t = time.time()
        with s.lock: s.ready += [(i, chunk, t) for i in ids]
    def push(s, z, chunk=-1):
        ev = torch.cuda.Event(); ev.record()
        with torch.cuda.stream(s.st):
            s.st.wait_event(ev); z.record_stream(s.st)
            with torch.no_grad(), _amp.autocast(dtype=torch.bfloat16 if a.vae_bf16_decode else s.v.dtype):
                sc = s.v.scale; zz = z.unsqueeze(0) / sc[1].view(1, -1, 1, 1, 1) + sc[0].view(1, -1, 1, 1, 1); x = s.m.conv2(zz)
                for i in range(x.shape[2]):
                    s.m._conv_idx = [0]; o = s.m.decoder(x[:, :, i:i + 1], feat_cache=s.m._feat_map, feat_idx=s.m._conv_idx).float().clamp_(-1, 1); s.out.append(o)
                    host = ((o[0] + 1) * 127.5).to(torch.uint8).to("cpu", non_blocking=True); e2 = torch.cuda.Event(); e2.record(s.st)
                    ids = list(range(s.nf, s.nf + o.shape[2])); s.nf += o.shape[2]; s.q.append(host)
                    th = threading.Thread(target=s._watch, args=(e2, ids, chunk)); th.start(); s.th.append(th)
    def finish(s):
        torch.cuda.current_stream().wait_stream(s.st); [t.join() for t in s.th]; im = torch.cat(s.out, 2)[0]; s.m.clear_cache(); return im
def _write(fn, im):
    v = ((im + 1) / 2).permute(1, 2, 3, 0).cpu().numpy(); imageio.mimwrite(fn, list((v * 255).astype("uint8")), fps=16, quality=8)
writers = []
for task in a.tasks.split(","):
    T = ALL[task]
    for seed in [int(x) for x in a.seeds.split(",")]:
        out = "%s/%s/%s_s%d" % (a.out, task, a.arm, seed)
        if os.path.exists(out + "/loop.json"): continue
        if RANK0: os.makedirs(out, exist_ok=True)
        sync(); t0 = time.time()
        cond = B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=a.max_area); B.reset_caches()
        if a.sp: FSP.local_head_caches(B)
        sync(); prep = time.time() - t0
        N = cond["num_chunks"]; sig = [float(s) for s in B.node_sigmas()]; g = torch.Generator().manual_seed(seed); tm = dict(denoise=0.0, commit=0.0); lat = []; chunk_start = []
        SD = StreamDecoder(B.pipe.vae) if (a.stream_decode and not a.no_video and RANK0 and not a.decode_rank) else None
        if a.decode_rank and RANK0: SH = B._chunk_shape; dist.send(torch.tensor([N, SH[0], SH[1], SH[2], SH[3], 0], dtype=torch.int64, device=DEV), dst=DEC); PEND = []
        sync(); t0 = time.time()
        with torch.no_grad():
            for ci in range(N):
                chunk_start.append(time.time()); z = B.chunk_noise(ci, g); t1 = time.time()
                for i in range(len(sig)):
                    x = B.target_x0(z, i, ci)
                    if i < len(sig) - 1: s_ = sig[i + 1]; z = (1 - s_) * x + s_ * B.randn_like(x, g)
                x = x.float(); sync(); tm["denoise"] += time.time() - t1; t1 = time.time(); B.commit(x, ci); sync(); tm["commit"] += time.time() - t1; lat.append(x.cpu())
                if SD is not None: SD.push(x, ci)
                if a.decode_rank and RANK0: xs = x.contiguous(); PEND.append((dist.isend(xs, dst=DEC), xs))
        sync(); gen = time.time() - t0; L = torch.cat(lat, 1); dec = None
        if not RANK0: continue
        torch.save(L.half(), out + "/lat.pt")
        if a.decode_rank:   # tell the decode worker where to write; it records its own frame-ready times
            [w.wait() for w, _ in PEND]; nb = out.encode()[:512]; dist.send(torch.tensor(list(nb) + [0] * (512 - len(nb)), dtype=torch.uint8, device=DEV), dst=DEC)
        exact = None
        if not a.no_video and not a.decode_rank:
            sync(); t1 = time.time()
            if SD is not None: im = SD.finish()
            else:
                with torch.no_grad(): im = B.pipe.vae.decode([L.cuda()])[0].float()
            sync(); dec = time.time() - t1
            if a.check_decode and SD is not None:
                with torch.no_grad(): im2 = B.pipe.vae.decode([L.cuda()])[0].float()
                exact = float((im - im2.clamp(-1, 1)).abs().max())
            th = threading.Thread(target=_write, args=(out + "/video.mp4", im.cpu())); th.start(); writers.append(th)
        e2e = prep + gen + (dec or 0)
        json.dump(dict(task=task, seed=seed, arm=a.arm, sp=(dist.get_world_size() if a.sp else 1), fp8=a.fp8, compile=a.compile, steps=list(TI), chunks=N, gen_s=round(gen, 2), prepare_s=round(prep, 2), decode_s=None if dec is None else round(dec, 2), e2e_s=round(e2e, 2), chunk_start=chunk_start, frame_ready=(sorted(SD.ready) if SD is not None else None), t_gen_start=t0, stream_decode=a.stream_decode, t5_gpu=a.t5_gpu, decode_maxdiff_vs_full=exact,
                       times={k: round(v, 2) for k, v in tm.items()}), open(out + "/loop.json", "w"), indent=1)
        print("[GEN_DONE] %s s%d %s gen %.1fs (denoise %.1f commit %.1f) prep %.1f dec(tail) %s e2e %.1f exact %s" % (task, seed, a.arm, gen, tm["denoise"], tm["commit"], prep, dec, e2e, exact), flush=True)
[t.join() for t in writers]; print("[ALL_DONE]", flush=True)
if a.sp:
    if a.decode_rank and RANK0: dist.send(torch.tensor([-1, 0, 0, 0, 0, 0], dtype=torch.int64, device=DEV), dst=DEC)
    dist.barrier(); dist.destroy_process_group()
