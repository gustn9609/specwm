"""Resolution split, end to end on one GPU (the device part is timed separately): cloud = 14B 2-step at 576x320 (FP8/Sage/compile/ring KV);
device = 1.3B + refiner LoRA (reslora_train.py): bilinear-upsample the cloud chunk to the 832x480 grid, re-noise to sigma_r, one forward -> 832 chunk,
commit it to the 1.3B's own KV, stream-decode at 832. Writes lat.pt (832 latents), video.mp4, loop.json with cloud / device times.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.res_split_infer --tasks a,b --seeds 1234 --lora specwm_out/reslora/r900/res_lora.pt --out ROOT [--arm NAME] [--no-refine]"""
import argparse, os, sys, json, time, torch, torch.nn.functional as F
if os.environ.get("RESERVE_GB"): import torch as _t; _r = _t.empty(int(os.environ["RESERVE_GB"]) << 30, dtype=_t.uint8, device="cuda:0"); del _r   # grab the GPU before other users' jobs do
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--tasks", required=True); ap.add_argument("--seeds", default="1234"); ap.add_argument("--lora", default=""); ap.add_argument("--out", required=True); ap.add_argument("--arm", default="")
ap.add_argument("--cloud-lora", default="", help="drift-reduction LoRA (drift_rft_train.py) merged into the 14B before the FP8 swap"); ap.add_argument("--lo-area", type=int, default=175104); ap.add_argument("--no-refine", action="store_true", help="device = plain bilinear latent upsample (no 1.3B forward)"); ap.add_argument("--no-video", action="store_true"); ap.add_argument("--warm", type=int, default=1)
ap.add_argument("--switch-tau", type=float, default=None, help="scene-adaptive: score every cloud commit with the drift head; once a chunk logit exceeds this, rebuild the 14B K/V at 832x480 from the device's refined 832 chunks (sink + window only) and run the rest of the clip at 832 (no refiner)")
ap.add_argument("--fused", action="store_true", help="fused commit: keep the K/V the last cloud denoising forward wrote instead of a separate clean commit pass"); ap.add_argument("--cloud-nodes", default="0,500"); ap.add_argument("--refresh", type=int, default=0, help="with --fused: run a clean commit pass on every K-th chunk (c %% K == K-1) to stop drift building up")
ap.add_argument("--partial-commit", type=int, default=0, help="with --fused: re-commit only the last K latent frames of the chunk with a clean t=0 pass (token subset, ~K/3 of a forward) to remove the chunk-boundary jump")
ap.add_argument("--stream-decode", type=int, default=0, help="1: decode every cloud chunk right after it is generated (side CUDA stream, separate VAE instance) and record per-frame ready times (sdv2_latency / wm_latency.py format); writes <out>/stream.mp4"); ap.add_argument("--no-device", action="store_true", help="the device does nothing but decode (no 1.3B: no prepare, no refiner, no commit)"); ap.add_argument("--frames", type=int, default=161, help="video frames (StEvo-Bench camera tasks: 253)"); ap.add_argument("--save-cloud", action="store_true", help="also decode the cloud's own low-res latents (no refiner) to <out>/cloud/spec.mp4")
ap.add_argument("--head", default=S + "/lingbot-world-v2/specwm_out/drift_head/head_L20_all.pt")
ap.add_argument("--hi-first", type=int, default=0, help="anchor test: generate the first K chunks with the cloud at 832x480 (no refiner), then rebuild the 14B K/V at the low resolution from those chunks (latents resampled) and continue low-res + refiner")
a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"; dev = "cuda:0"
Tg = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=tuple(int(v) for v in a.cloud_nodes.split(",")))
st = torch.load(a.lora, map_location="cpu") if a.lora else None; RI = (st["ref_index"] if isinstance(st["ref_index"], list) else [st["ref_index"]]) if st else [900]
D = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=tuple([0] + RI), cfg_name="i2v-1.3B", assets_dir=FAST)
if st is not None:   # merge W += B A (block order q,k,v,o[,ffn0,ffn2])
    it = iter(zip(st["A"], st["B"]))
    with torch.no_grad():
        for blk in D.target_model.blocks:
            for lin in [getattr(blk.self_attn, n) for n in ("q", "k", "v", "o")] + ([blk.ffn[0], blk.ffn[2]] if st["ffn"] else []): A_, B_ = next(it); lin.weight += (B_.float() @ A_.float()).to(lin.weight.device, lin.weight.dtype)
    print("[res_split] merged refiner LoRA", a.lora, "step", st.get("step"), "sigma_r %.3f" % float(D.node_sigmas()[1]), flush=True)
if a.cloud_lora:
    cs = torch.load(a.cloud_lora, map_location="cpu"); it = iter(zip(cs["A"], cs["B"]))
    with torch.no_grad():
        for blk in Tg.target_model.blocks:
            for lin in [getattr(blk.self_attn, n) for n in ("q", "k", "v", "o")] + ([blk.ffn[0], blk.ffn[2]] if cs["ffn"] else []): A_, B_ = next(it); lin.weight += (B_.float() @ A_.float()).to(lin.weight.device, lin.weight.dtype)
    print("[res_split] merged cloud LoRA", a.cloud_lora, "step", cs.get("step"), flush=True)
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(Tg.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 512; _dyn.config.accumulated_recompile_limit = 8192   # 576 + 832 shapes x KV-window lengths x 2 aspect ratios
for blk in Tg.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
for blk in D.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
import torch.nn as nn
def _cache_text(B):
    enc = B.pipe.text_encoder; cache = {}
    class _C:
        def __init__(s): s.model = enc.model
        def __call__(s, texts, device):
            k = (tuple(texts), str(device))
            if k not in cache: cache[k] = enc(texts, device)
            return cache[k]
    B.pipe.text_encoder = _C()
_cache_text(Tg); _cache_text(D)
H = {}
if a.switch_tau is not None:
    class Head(nn.Module):
        def __init__(s, C, hid):
            super().__init__(); s.mu = nn.Parameter(torch.zeros(C), requires_grad=False); s.sd = nn.Parameter(torch.ones(C), requires_grad=False)
            s.tok = nn.Sequential(nn.Linear(C, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU()); s.pos = nn.Parameter(torch.zeros(16, hid)); s.out = nn.Sequential(nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))
        def chunk_logits(s, x): h = s.tok((x - s.mu) / s.sd) + s.pos; return s.out(h.mean(1)).squeeze(-1)
    hk = torch.load(a.head, map_location="cpu"); HEAD = Head(5120, hk["hid"]).to(dev); HEAD.load_state_dict(hk["state"]); HEAD.eval()
    Tg.target_model.blocks[19].register_forward_hook(lambda m, i, o: H.__setitem__("h", o[0] if isinstance(o, tuple) else o))
def head_logit(grid):
    x = H["h"][0].float().view(*grid, -1).permute(3, 0, 1, 2); g = F.adaptive_avg_pool3d(x, (1, 3, 5)).reshape(x.shape[0], 15).t()
    return float(HEAD.chunk_logits(torch.cat([x.mean((1, 2, 3))[None], g])[None])[0])
SIG = [float(s) for s in Tg.node_sigmas()]; SIGR = [float(s) for s in D.node_sigmas()]; SR = SIGR[1]; arm = a.arm or ("hifirst%d_" % a.hi_first if a.hi_first else "") + (("ressplit_r%s" % "_".join(map(str, RI)) if not a.no_refine else "ressplit_bilinear") + ("_switch%g" % a.switch_tau if a.switch_tau is not None else "")); sync = torch.cuda.synchronize
import threading, imageio
from torch.cuda import amp as _amp
class StreamDecoder:   # as in gen_fast.py: per-latent-frame causal decode on a side stream; ready[i] = (pixel_frame_id, chunk, wall time in host memory)
    def __init__(s, vae): s.v = vae; s.m = vae.model; s.m.clear_cache(); s.st = torch.cuda.Stream(); s.out = []; s.ready = []; s.nf = 0; s.q = []; s.lock = threading.Lock(); s.th = []
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
                    s.m._conv_idx = [0]; o = s.m.decoder(x[:, :, i:i + 1], feat_cache=s.m._feat_map, feat_idx=s.m._conv_idx).float().clamp_(-1, 1); s.out.append(o)
                    host = ((o[0] + 1) * 127.5).to(torch.uint8).to("cpu", non_blocking=True); e2 = torch.cuda.Event(); e2.record(s.st)
                    ids = list(range(s.nf, s.nf + o.shape[2])); s.nf += o.shape[2]; s.q.append(host)
                    th = threading.Thread(target=s._watch, args=(e2, ids, chunk)); th.start(); s.th.append(th)
    def finish(s):
        torch.cuda.current_stream().wait_stream(s.st); [t.join() for t in s.th]; im = torch.cat(s.out, 2)[0]; s.m.clear_cache(); return im
VAE2 = None
if a.stream_decode:
    from wan.modules.vae2_1 import Wan2_1_VAE
    VAE2 = Wan2_1_VAE(vae_pth=FAST + "/Wan2.1_VAE.pth", device="cuda:0", dtype=Tg.pipe.vae.dtype)
def gen(seed, c): return torch.Generator().manual_seed(seed * 1009 + c * 7)
def genr(seed, c): return torch.Generator().manual_seed(seed * 7919 + c * 13 + 5)
@torch.no_grad()
def run(task, seed, out):
    T = ALL[task]; t0 = time.time()
    cond = Tg.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=a.frames, max_area=a.lo_area); Tg.reset_caches(); N = cond["num_chunks"]
    if not a.no_device: D.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=a.frames, max_area=480 * 832); D.reset_caches(); HW = tuple(D._chunk_shape[2:])
    sync(); prep = time.time() - t0; CS = []; SD = StreamDecoder(VAE2) if VAE2 is not None else None
    chunks = []; tc = td = tsw = 0.0; tg = time.time(); logits = []; switched_at = None; hi = False; c_cloud = []; c_dev = []; cl = []
    def grid(): return (3, Tg._chunk_shape[2] // 2, Tg._chunk_shape[3] // 2)
    if a.hi_first > 0:   # start the cloud at 832 for the first K chunks
        Tg.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=a.frames, max_area=480 * 832); Tg.reset_caches(); hi = True
    for c in range(N):
        CS.append(time.time())
        if a.hi_first > 0 and c == a.hi_first and hi:   # step down to the low resolution: rebuild the cloud K/V from the 832 chunks
            sync(); t3 = time.time(); Tg.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=a.frames, max_area=a.lo_area); Tg.reset_caches()
            for cc in sorted({0, 1} | set(range(max(2, c - 6), c))):
                if cc >= c: continue
                xc = F.interpolate(chunks[cc].to(dev).float().permute(1, 0, 2, 3), size=tuple(Tg._chunk_shape[2:]), mode="bilinear", align_corners=False, antialias=True).permute(1, 0, 2, 3); Tg.commit(xc, cc)
            Tg._committed_frames = c * Tg._chunk_size; sync(); tsw += time.time() - t3; tc += time.time() - t3; hi = False
        g = gen(seed, c); z = Tg.chunk_noise(c, g); sync(); t1 = time.time()
        for i in range(len(SIG)):
            if i > 0: s_ = SIG[i]; z = (1 - s_) * x + s_ * Tg.randn_like(x, g)
            x = Tg.target_x0(z, i, c).float()
        if a.fused and a.partial_commit and not (a.refresh and c % a.refresh == a.refresh - 1):   # clean K/V only for the newest latent frames
            fs = Tg._frame_seqlen; L = Tg._chunk_size * fs; FP.ACTIVE["a"] = dict(idx=torch.arange(L - a.partial_commit * fs, L, device=dev), n=a.partial_commit * fs, L=L)
            try: Tg.commit(x, c)
            finally: FP.ACTIVE["a"] = None
        elif a.fused and not (a.refresh and c % a.refresh == a.refresh - 1): Tg._committed_frames += Tg._chunk_size
        else: Tg.commit(x, c)
        sync(); tc += time.time() - t1; c_cloud.append(round(time.time() - t1, 4)); t2 = time.time()
        if a.save_cloud: cl.append(x.half().cpu())
        if SD is not None: SD.push(x, c)
        if a.no_device: chunks.append(x.cpu()); c_dev.append(0.0); logits.append(None); continue
        lg = head_logit(grid()) if a.switch_tau is not None else None; logits.append(None if lg is None else round(lg, 2))
        if hi:   # already switched: the cloud chunk is 832 already; the device only keeps its K/V in sync
            xh = x if tuple(x.shape[2:]) == HW else F.interpolate(x.permute(1, 0, 2, 3), size=HW, mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
        else:
            up = F.interpolate(x.permute(1, 0, 2, 3), size=HW, mode="bilinear", align_corners=False).permute(1, 0, 2, 3)   # cloud latent -> 832 grid
            if a.no_refine: xh = up
            else:
                gr = genr(seed, c); xh = up
                for k in range(1, len(SIGR)): zr = (1 - SIGR[k]) * xh + SIGR[k] * D.randn_like(up, gr); xh = D.target_x0(zr, k, c).float()
        D.commit(xh, c); sync(); td += time.time() - t2; c_dev.append(round(time.time() - t2, 4)); chunks.append(xh.cpu())
        if a.switch_tau is not None and not hi and c >= 1 and c < N - 1 and lg > a.switch_tau:   # upgrade the cloud to 832 for the rest of the clip
            sync(); t3 = time.time(); Tg.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=a.frames, max_area=480 * 832); Tg.reset_caches()
            keep = sorted({0, 1} | set(range(max(2, c - 5), c + 1)))   # sink chunks + the local window = what the ring cache holds
            for cc in keep:
                xc = chunks[cc].to(dev).float()
                if tuple(xc.shape[2:]) != tuple(Tg._chunk_shape[2:]): xc = F.interpolate(xc.permute(1, 0, 2, 3), size=tuple(Tg._chunk_shape[2:]), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
                Tg.commit(xc, cc)
            Tg._committed_frames = (c + 1) * Tg._chunk_size; sync(); tsw += time.time() - t3; tc += time.time() - t3; hi = True; switched_at = c
    gen_s = time.time() - tg; os.makedirs(out, exist_ok=True); lat = torch.cat(chunks, 1); dec = 0.0
    if os.environ.get("SAVE_LAT", "1") != "0": torch.save(lat.half(), out + "/lat.pt")
    fr_ready = None
    if SD is not None:
        t3 = time.time(); im = SD.finish(); sync(); dec = time.time() - t3; fr_ready = sorted(SD.ready)
        if os.environ.get("NO_MP4") != "1": v = ((im.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).float().cpu().numpy(); imageio.mimwrite(out + "/stream.mp4", list((v * 255).astype("uint8")), fps=16, quality=8)
    elif not a.no_video and not a.no_device: t3 = time.time(); D.decode_and_save([lat.to(dev)], out); dec = time.time() - t3
    if a.save_cloud and cl:
        os.makedirs(out + "/cloud", exist_ok=True); cz = torch.cat(cl, 1)
        if os.environ.get("SAVE_LAT", "1") != "0": torch.save(cz, out + "/cloud/lat.pt")
        Tg.decode_and_save([cz.to(dev).float()], out + "/cloud")
    R = dict(task=task, seed=seed, arm=arm, chunks=N, prepare_s=round(prep, 2), gen_s=round(gen_s, 2), cloud_s=round(tc, 2), device_s=round(td, 2), switch_s=round(tsw, 2), switched_at=switched_at, cloud_chunk_s=c_cloud, device_chunk_s=c_dev, chunk_start=CS, frame_ready=fr_ready, t_gen_start=tg, stream_decode=a.stream_decode, no_device=a.no_device, logits=logits, lo_area=a.lo_area, lo_shape=list(Tg._chunk_shape) if switched_at is None else None, decode_s=round(dec, 2), e2e_s=round(time.time() - t0, 2), sigma_r=SR, lora=a.lora)
    json.dump(R, open(out + "/loop.json", "w"), indent=1); print("[RES_SPLIT] %s s%d %s gen %.1fs (cloud %.1f device %.1f switch %.1f at %s) e2e %.1f | logits %s" % (task, seed, arm, gen_s, tc, td, tsw, switched_at, R["e2e_s"], logits), flush=True)
tasks = a.tasks.split(","); seeds = [int(s) for s in a.seeds.split(",")]
if a.warm: run(tasks[0], seeds[0], a.out + "/%s/%s_warm" % (tasks[0], arm))
for t in tasks:
    for s in seeds:
        if os.path.exists(a.out + "/%s/%s_s%d/loop.json" % (t, arm, s)): print("[skip]", t, s, flush=True); continue
        run(t, s, a.out + "/%s/%s_s%d" % (t, arm, s))
print("[ALL_DONE]", flush=True)
