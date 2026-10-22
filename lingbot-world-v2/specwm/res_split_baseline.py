"""[10/20] Resolution-split collaboration baseline inside the frozen baseline protocol (baselines/PROTOCOL.md): the advisor's "strongest
collaboration baseline" = send the actions to the cloud, the cloud generates at LOW resolution, send it back, the device upsamples.
Port of specwm/res_split_infer.py (its default, non-fused loop; the switch / hi-first / fused / partial-commit options are NOT carried over)
that (a) takes the frozen baseline task dicts (baselines/tasks/<SET>.json or a gen/ chunk json {"tasks": [...]}: id, image, camera_WM, action dir
with poses.npy + intrinsics.npy, area, frames) instead of data/stevo/tasks_camera.json, (b) streams the DEVICE's 832x480 latents through a side-stream
VAE (TTFF / e2e measurable, as two_machine --timed-decode 1; res_split_infer streamed the cloud's low-res latents), and (c) writes the frozen layout
baselines/runs/ours_<ARM>_<SET>/<task>/{video.mp4 (161 frames), timing.json} in the SAME schema as baselines/ours_to_timing.py (its helpers are imported,
nothing there is edited).
Arms:  rs576  = cloud 14B FP8 2-step (nodes 0,500 = t [1000, 500]) at 576x320, device = bilinear latent upsample to the 832x480 latent grid (no model)
       rs576r = the same cloud + device 1.3B causal-fast (bf16) with a merged refiner LoRA (reslora_train.py): re-noise the upsampled chunk to the
                LoRA's sigma(s), one 1.3B forward each, commit to the 1.3B's own KV window.  --lora is REQUIRED for rs576r (specwm_out/reslora/r900/
                res_lora.pt was purged; r850_930_pix/res_lora.pt (ref_index [850, 930]) is the only one left on 10/20).
Both parts run SERIALLY on ONE GPU (one process, no network): per chunk  chunk_s = cloud_chunk_s + device_chunk_s  (facts.md §12: the listed 9.8 /
15.8 fps assumed a parallel device; here both lists are recorded and steady_fps_gen uses the serial sum; steady_fps_cloud_only = the parallel bound).
The streamed 832x480 decode of chunk c runs on a side stream and the sync at the start of chunk c+1 waits for it (DiT and VAE sequential, like the
baselines' e2e): it is outside chunk_s and inside ttff_s / e2e_s / steady_fps_e2e.  spec.mp4 (the frozen video) is the usual full VAE decode of the
device latents after the clip, trimmed to 161 frames by ours_to_timing.trim (libx264 crf 18).
Low resolution: V2Binding.prepare's floor formula at the 832x480 aspect gives 576x320 only for max_area in [191424, 195680] (default --lo-area 193536);
res_split_infer's default 175104 gives 544x304 on an 832x480 image (it was 560x304 on the 1408x768 StEvo images).  Only the 832x480 sets are
supported (area 399361: neutral_832, neutral_832_nocap, astra); any other area is refused.
usage: CUDA_VISIBLE_DEVICES=g /home/hk28965/specwm/scripts/envrun.sh python -m specwm.res_split_baseline --tasks-json J --tasks a,b --arm rs576|rs576r
         --set SET --out RAW_DIR --runs FROZEN_DIR [--area 399361] [--lo-area 193536] [--lora F] [--seed 1234] [--warm 1] [--frames 165] [--force 0]
Skip-if-done per task (FROZEN_DIR/<id>/timing.json). Raw: RAW_DIR/<id>/loop.json (+ spec.mp4 until converted, lat.pt only with SAVE_LAT=1).
Caller = baselines/gen/run_ressplit.sh (takes no lock itself; the queue chunk takes flock /tmp/gpu<G>.lock)."""
import argparse, os, sys, json, time, threading
ap = argparse.ArgumentParser()
ap.add_argument("--tasks-json", required=True, help="task table: a plain list or {'tasks': [...]} (gen/ chunk json)"); ap.add_argument("--tasks", default="", help="comma list of task ids (default: every task of --tasks-json)")
ap.add_argument("--arm", required=True, choices=["rs576", "rs576r"]); ap.add_argument("--set", default=None); ap.add_argument("--out", required=True, help="raw dir"); ap.add_argument("--runs", required=True, help="frozen runs dir (runs/ours_<ARM>_<SET>)")
ap.add_argument("--area", type=int, default=399361, help="output max_area (the set's; 399361 = exact 832x480)"); ap.add_argument("--lo-area", type=int, default=193536, help="cloud max_area (193536 -> 576x320 at the 832x480 aspect)")
ap.add_argument("--lora", default="", help="rs576r: refiner LoRA (reslora_train.py state: A, B, ffn, ref_index, step, rank)"); ap.add_argument("--cloud-nodes", default="0,500"); ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--warm", type=int, default=1, help="1: compile warm-up on the first task (seed + 1), nothing written"); ap.add_argument("--frames", type=int, default=0, help="generated frames (0 = the task's 'frames', 165)"); ap.add_argument("--force", type=int, default=0)
ap.add_argument("--video", type=int, default=1, help="0: no full decode / conversion (timing only; nothing frozen)")
a = ap.parse_args()
S = "/home/hk28965/specwm"; B = S + "/baselines"; sys.path.insert(0, S + "/lingbot-world-v2"); sys.path.insert(0, B); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
os.environ["PATH"] = os.path.expanduser("~/.local/bin") + ":" + S + "/tools/ffmpeg-7.0.2-amd64-static:" + os.environ.get("PATH", "")   # ffmpeg / ffprobe for ours_to_timing.trim / probe
import torch, torch.nn.functional as F, numpy as np, imageio
import ours_to_timing as OT   # [10/20] frozen-schema helpers (probe, trim, e2e_from_ready, median, CKPT / PREC / ATTN strings); the file itself is not edited
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
TJ = json.load(open(a.tasks_json)); TJ = TJ["tasks"] if isinstance(TJ, dict) else TJ; ALL = {t["id"]: t for t in TJ}
TASKS = [t for t in a.tasks.split(",") if t] or [t["id"] for t in TJ]
for t in TASKS: assert t in ALL, "task %s not in %s" % (t, a.tasks_json)
areas = set(int(ALL[t].get("area", a.area)) for t in TASKS); assert areas == {a.area}, ("the tasks' area must equal --area", areas, a.area)
if a.area != 399361: print("[res_split_baseline] REFUSED: arms rs576 / rs576r support only the 832x480 sets (area 399361: neutral_832, neutral_832_nocap, astra); got area %d (the cloud low resolution for other aspects is not defined yet)" % a.area, flush=True); sys.exit(1)
if a.arm == "rs576r" and not a.lora: print("[res_split_baseline] rs576r needs --lora (specwm_out/reslora/<run>/res_lora.pt)", flush=True); sys.exit(1)
REFINE = a.arm == "rs576r"; dev = "cuda:0"; GiB = 2 ** 30; sync = torch.cuda.synchronize; now = time.time; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
os.makedirs(a.out, exist_ok=True); os.makedirs(a.runs, exist_ok=True)
def frames_of(t): return int(a.frames or ALL[t].get("frames", 165))
def lat_hw(image_path, max_area):
    """V2Binding.prepare's floor formula (vae stride 8, patch 2): (lat_h, lat_w) for this image's aspect"""
    from PIL import Image
    w, h = Image.open(image_path).size; asp = h / w
    return int(round(np.sqrt(max_area * asp) // 8 // 2 * 2)), int(round(np.sqrt(max_area / asp) // 8 // 2 * 2))
# ---------------- models (as res_split_infer.py) ----------------
CN = tuple(int(v) for v in a.cloud_nodes.split(","))
Tg = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=CN)
D = None; st = None; RI = []
if REFINE:
    st = torch.load(a.lora, map_location="cpu"); RI = st["ref_index"] if isinstance(st["ref_index"], list) else [st["ref_index"]]
    D = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=tuple([0] + list(RI)), cfg_name="i2v-1.3B", assets_dir=FAST)
    it = iter(zip(st["A"], st["B"]))   # merge W += B A (block order q,k,v,o[,ffn0,ffn2])
    with torch.no_grad():
        for blk in D.target_model.blocks:
            for lin in [getattr(blk.self_attn, n) for n in ("q", "k", "v", "o")] + ([blk.ffn[0], blk.ffn[2]] if st["ffn"] else []): A_, B_ = next(it); lin.weight += (B_.float() @ A_.float()).to(lin.weight.device, lin.weight.dtype)
    print("[res_split_baseline] merged refiner LoRA %s step %s rank %s ffn %s ref_index %s sigmas %s" % (a.lora, st.get("step"), st.get("rank"), st.get("ffn"), RI, [round(float(s_), 4) for s_ in D.node_sigmas()[1:]]), flush=True)
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(Tg.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 512; _dyn.config.accumulated_recompile_limit = 8192
for blk in Tg.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
if D is not None:
    for blk in D.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
def _cache_text(Bd):
    enc = Bd.pipe.text_encoder; cache = {}
    class _C:
        def __init__(s): s.model = enc.model
        def __call__(s, texts, device):
            k = (tuple(texts), str(device))
            if k not in cache: cache[k] = enc(texts, device)
            return cache[k]
    Bd.pipe.text_encoder = _C()
_cache_text(Tg)
if D is not None: _cache_text(D)
SIG = [float(s) for s in Tg.node_sigmas()]; SIGR = [float(s) for s in D.node_sigmas()] if D is not None else [1.0]
from wan.modules.vae2_1 import Wan2_1_VAE
VAE2 = Wan2_1_VAE(vae_pth=FAST + "/Wan2.1_VAE.pth", device=dev, dtype=Tg.pipe.vae.dtype)   # side-stream decoder (second VAE instance, as two_machine --timed-decode 1)
class StreamDecoder:   # = two_machine.StreamDecoder (frames kept on the host as uint8 only; ready[i] = (frame, chunk, wall time))
    def __init__(s, vae): s.v = vae; s.m = vae.model; s.m.clear_cache(); s.st = torch.cuda.Stream(); s.ready = []; s.nf = 0; s.lock = threading.Lock(); s.th = []; s.ev = []; s.host = []
    def _watch(s, ev, ids, chunk):
        ev.synchronize(); t = now()
        with s.lock: s.ready += [(i, chunk, t) for i in ids]
    def push(s, z, chunk=-1):
        ev = torch.cuda.Event(); ev.record(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); tp = now()
        with torch.cuda.stream(s.st):
            s.st.wait_event(ev); z.record_stream(s.st); e0.record(s.st)
            with torch.no_grad(), torch.autocast("cuda", dtype=s.v.dtype):
                sc = s.v.scale; zz = z.unsqueeze(0) / sc[1].view(1, -1, 1, 1, 1) + sc[0].view(1, -1, 1, 1, 1); x = s.m.conv2(zz)
                for i in range(x.shape[2]):
                    s.m._conv_idx = [0]; o = s.m.decoder(x[:, :, i:i + 1], feat_cache=s.m._feat_map, feat_idx=s.m._conv_idx).float().clamp_(-1, 1)
                    host = ((o[0] + 1) * 127.5).to(torch.uint8).to("cpu", non_blocking=True); e2 = torch.cuda.Event(); e2.record(s.st)
                    ids = list(range(s.nf, s.nf + o.shape[2])); s.nf += o.shape[2]; s.host.append(host)
                    th = threading.Thread(target=s._watch, args=(e2, ids, chunk)); th.start(); s.th.append(th)
            e1.record(s.st)
        s.ev.append((chunk, tp, e0, e1))
    def finish(s):
        torch.cuda.current_stream().wait_stream(s.st); sync(); [t.join() for t in s.th]; s.m.clear_cache()
        return [dict(c=c, push_t=tp, gpu_ms=round(e0.elapsed_time(e1), 2)) for c, tp, e0, e1 in s.ev]
def gen(seed, c): return torch.Generator().manual_seed(seed * 1009 + c * 7)
def genr(seed, c): return torch.Generator().manual_seed(seed * 7919 + c * 13 + 5)
LO_HW = None   # (lat_h, lat_w) of the cloud, checked once
# ---------------- one clip ----------------
@torch.no_grad()
def run(task, seed, od, warm=False):
    global LO_HW
    T = ALL[task]; nfr = frames_of(task); os.makedirs(od, exist_ok=True); torch.cuda.reset_peak_memory_stats(); sync(); tp0 = now()
    cond = Tg.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=nfr, max_area=a.lo_area); Tg.reset_caches(); N = cond["num_chunks"]; lo = tuple(Tg._chunk_shape[2:])
    if D is not None: D.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=nfr, max_area=a.area); D.reset_caches(); HW = tuple(D._chunk_shape[2:])
    else: HW = lat_hw(T["image"], a.area)
    assert HW == (60, 104), ("output latent grid is not 832x480", HW)
    if a.lo_area == 193536: assert lo == (40, 72), ("cloud latent grid is not 576x320 at --lo-area 193536", lo)
    if LO_HW is None: print("[res_split_baseline] cloud latent grid %s = %dx%d px, device grid %s = %dx%d px, %d chunks" % (lo, lo[1] * 8, lo[0] * 8, HW, HW[1] * 8, HW[0] * 8, N), flush=True)
    LO_HW = lo
    sync(); prep_s = now() - tp0; SD = StreamDecoder(VAE2); chunks = []; tc = []; cc = []; dd = []; CS = []; t0 = now()
    for c in range(N):
        sync(); t1 = now(); CS.append(t1)   # the sync waits for the side-stream decode of chunk c-1 (sequential DiT + VAE, as two_machine solo)
        g = gen(seed, c); z = Tg.chunk_noise(c, g); x = None
        for i in range(len(SIG)):
            if i > 0: s_ = SIG[i]; z = (1 - s_) * x + s_ * Tg.randn_like(x, g)
            x = Tg.target_x0(z, i, c).float()
        Tg.commit(x, c); sync(); t2 = now(); cc.append(round(t2 - t1, 4))   # cloud: 2-step denoise + clean commit at 576x320
        xh = F.interpolate(x.permute(1, 0, 2, 3), size=HW, mode="bilinear", align_corners=False).permute(1, 0, 2, 3)   # device: cloud latent -> 832x480 grid
        if D is not None:
            gr = genr(seed, c)
            for k in range(1, len(SIGR)): zr = (1 - SIGR[k]) * xh + SIGR[k] * D.randn_like(xh, gr); xh = D.target_x0(zr, k, c).float()
            D.commit(xh, c)
        xh = xh.contiguous(); sync(); t3 = now(); dd.append(round(t3 - t2, 4)); tc.append(round(t3 - t1, 4)); chunks.append(xh)
        SD.push(xh.float(), c)
    t1 = now(); cd = SD.finish(); fr = sorted(SD.ready); last = {}; fin_wait = now() - t1
    for f_, c_, t_ in fr: last[c_] = max(last.get(c_, 0.0), t_)
    for r_ in cd: r_["ready_t"] = last.get(r_["c"]); r_["wall_s"] = round(last[r_["c"]] - r_["push_t"], 4) if r_["c"] in last else None
    pk_a = round(torch.cuda.max_memory_allocated() / GiB, 3); pk_r = round(torch.cuda.max_memory_reserved() / GiB, 3)
    ttff, e2e, fps_e2e, iv = OT.e2e_from_ready(fr, prep_s, t0)
    st_ = tc[3:] if len(tc) > 4 else tc; stc = cc[3:] if len(cc) > 4 else cc; std = dd[3:] if len(dd) > 4 else dd
    L = dict(task=task, seed=seed, arm=a.arm, set=a.set, chunks=N, frames_arg=nfr, prep_s=round(prep_s, 4), t_gen_start=t0, chunk_start=CS, chunk_s=tc, cloud_chunk_s=cc, device_chunk_s=dd, gen_s=round(sum(tc), 4), loop_wall_s=round(t1 - t0, 4),
             cloud_s=round(sum(cc), 4), device_s=round(sum(dd), 4), steady_chunk_s=OT.median(st_), steady_cloud_chunk_s=OT.median(stc), steady_device_chunk_s=OT.median(std), fps=round(12.0 / OT.median(st_), 2),
             timed_decode=1, frame_ready=fr, frames_streamed=SD.nf, chunk_decode=cd, decode_s=round(sum(r_["gpu_ms"] for r_ in cd) / 1000.0, 4), finish_wait_s=round(fin_wait, 4), ttff_s=ttff, e2e_s=e2e, steady_fps_e2e=fps_e2e, chunk_ready_interval_s=iv,
             peak_alloc_GiB=pk_a, peak_reserved_GiB=pk_r, lo_area=a.lo_area, lo_shape=list(Tg._chunk_shape), lo_resolution="%dx%d" % (lo[1] * 8, lo[0] * 8), hi_shape=[16, 3, HW[0], HW[1]], cloud_nodes=list(CN), cloud_sigmas=SIG, refine=REFINE,
             lora=(a.lora if REFINE else None), lora_meta=({k: st[k] for k in ("step", "rank", "ffn", "ref_index") if k in st} if st is not None else None), refiner_sigmas=(SIGR[1:] if REFINE else None), warm=warm)
    dec_full = None
    if not warm:
        lat = torch.cat(chunks, 1)
        if os.environ.get("SAVE_LAT", "0") == "1": torch.save(lat.half().cpu(), od + "/lat.pt")
        if a.video: t4 = now(); Tg.decode_and_save([lat.to(dev)], od); sync(); dec_full = round(now() - t4, 3)   # spec.mp4 = full VAE decode of the device latents (same call as two_machine solo)
        L["decode_full_s"] = dec_full; L["peak_alloc_proc_GiB"] = round(torch.cuda.max_memory_allocated() / GiB, 3)
    json.dump(L, open(od + "/loop.json", "w"), indent=1)
    print("[RS] %s s%d %s%s: prep %.1f gen %.1f (cloud %.1f device %.1f) steady %.3f s/chunk (cloud %.3f + device %.3f) -> %.1f fps serial / %.1f cloud-only; ttff %s e2e %s fps_e2e %s; peak %.1f / %.1f GiB; full decode %s" % (
        task, seed, a.arm, " WARM" if warm else "", prep_s, sum(tc), sum(cc), sum(dd), OT.median(st_), OT.median(stc), OT.median(std), 12.0 / OT.median(st_), 12.0 / OT.median(stc), ttff, e2e, fps_e2e, pk_a, pk_r, dec_full), flush=True)
    return L
# ---------------- frozen conversion (schema of ours_to_timing.convert, two_machine-solo branch) ----------------
MODEL = {"rs576": "Resolution-split collaboration baseline (cloud low-res + device upsample; advisor 10/20): CLOUD = LingBot-World 2.0 14B causal-fast, single-GPU FP8 port, realtime 2-step (nodes t=[1000, 500], separate clean commit) at 576x320 (max_area 193536, latent 40x72, its own KV window 18); DEVICE = bilinear upsample of every cloud chunk latent to the 832x480 latent grid (60x104), no model, decoded at 832x480 by the Wan VAE; both parts serial in ONE process on ONE GPU, no network (specwm.res_split_baseline, ported from res_split_infer.py --no-refine)",
         "rs576r": "Resolution-split collaboration baseline (cloud low-res + device upsample + refiner; advisor 10/20): CLOUD = LingBot-World 2.0 14B causal-fast, single-GPU FP8 port, realtime 2-step (nodes t=[1000, 500], separate clean commit) at 576x320 (max_area 193536, latent 40x72, its own KV window 18); DEVICE = bilinear upsample of every cloud chunk latent to the 832x480 latent grid (60x104) + LingBot-World 2.0 1.3B causal-fast (bf16) with a merged refiner LoRA (specwm/reslora_train.py; the chunk is re-noised to the LoRA's sigma(s) and one 1.3B forward each returns x0), committed to the 1.3B's own KV window 18, decoded at 832x480 by the Wan VAE; both parts serial in ONE process on ONE GPU, no network (specwm.res_split_baseline, ported from res_split_infer.py)"}
def convert(task, L, od, out):
    os.makedirs(out, exist_ok=True); nulls = {}; src = od + "/spec.mp4"; assert os.path.exists(src), "no video at " + src
    nout = OT.frames_out_for(task); w0, h0, n0, _ = OT.probe(src); dst = out + "/video.mp4"; OT.trim(src, dst, nout); w, h, n, fr = OT.probe(dst)
    assert n == min(nout, n0) and (w, h) == (w0, h0), (src, (w0, h0, n0), dst, (w, h, n))
    tc = L["chunk_s"]; cc = L["cloud_chunk_s"]; dd = L["device_chunk_s"]; st_ = tc[3:] if len(tc) > 4 else tc; stc = cc[3:] if len(cc) > 4 else cc; std = dd[3:] if len(dd) > 4 else dd
    steps = "cloud 2 (t=[1000, 500]) at 576x320; device 0 (bilinear latent upsample)" if not REFINE else "cloud 2 (t=[1000, 500]) at 576x320; device %d refiner forward(s) of the 1.3B at sigma %s (one per LoRA node) at 832x480" % (len(SIGR) - 1, [round(s_, 3) for s_ in SIGR[1:]])
    T = dict(model=MODEL[a.arm], ckpt=OT.CKPT14 + ("; device: " + OT.CKPT1P3 + "; refiner LoRA %s (%s)" % (a.lora, json.dumps(L["lora_meta"])) if REFINE else ""), precision=OT.PREC["fp8"] + (" (cloud 14B); device 1.3B: " + OT.PREC["bf16"] if REFINE else " (cloud 14B)"), attention=OT.ATTN + (" (both models)" if REFINE else ""),
             vae="Wan2.1_VAE.pth, full decode after generation (spec.mp4) of the DEVICE 832x480 latents; second VAE instance decodes every device chunk on a side stream right after its commit (ttff / e2e / decode_s), as two_machine --timed-decode 1",
             resolution_out="%dx%d" % (w, h), frames=n, frames_generated=n0, fps_native=16, frames_per_chunk=OT.FPC, frames_per_call_or_chunk="12 (chunk = 3 latent frames; chunk 0 yields 9 = 1 + 4 + 4)", steps=steps,
             gen_s=round(sum(tc), 3), e2e_s=L["e2e_s"], ttff_s=L["ttff_s"], steady_fps_gen=round(OT.FPC / OT.median(st_), 2), steady_fps_e2e=L["steady_fps_e2e"], peak_alloc_GiB=L["peak_alloc_GiB"], peak_reserved_GiB=L["peak_reserved_GiB"],
             text_encoder_resident=False, text_encoder="umT5-XXL on the CPU (V2Binding, encoded during prepare, cached per prompt string)" + ("; both bindings encode the prompt" if REFINE else ""), kv_window=18, kv_sink=6, wq="fp8", warm=True,
             chunk_s=tc, cloud_chunk_s=cc, device_chunk_s=dd, cloud_s=L["cloud_s"], device_s=L["device_s"], steady_chunk_s=OT.median(st_), steady_cloud_chunk_s=OT.median(stc), steady_device_chunk_s=OT.median(std),
             steady_fps_cloud_only=round(OT.FPC / OT.median(stc), 2), steady_fps_device_only=(round(OT.FPC / OT.median(std), 2) if OT.median(std) > 0 else None), loop_wall_s=L["loop_wall_s"],
             seed=L["seed"], task=task, nodes=",".join(str(v) for v in CN), use=",".join(str(i) for i in range(len(CN))), area=a.area, lo_area=a.lo_area, lo_resolution=L["lo_resolution"], lo_latent_hw=L["lo_shape"][2:], hi_latent_hw=L["hi_shape"][2:], frames_arg=L["frames_arg"],
             refiner=(dict(lora=a.lora, meta=L["lora_meta"], sigmas=L["refiner_sigmas"], model="1.3B causal-fast bf16, KV window 18 / sink 6") if REFINE else None),
             prepare_s=L["prep_s"], decode_s=L["decode_s"], decode_full_s=L.get("decode_full_s"), chunk_decode_s=[r_["gpu_ms"] / 1000.0 for r_ in L["chunk_decode"]], chunk_ready_interval_s=L["chunk_ready_interval_s"], frames_streamed=L["frames_streamed"],
             gpus=1, transport="none: cloud and device run in one process on one GPU (no RTT, no serialization); the 'send' is a device-side tensor copy",
             fps_note="chunk_s = cloud_chunk_s + device_chunk_s per chunk, measured SERIALLY on one GPU (cloud 2-step + commit at 576x320, then the device upsample%s + the device commit); steady_fps_gen = 12 / median chunk_s over chunks >= 3 (serial); steady_fps_cloud_only = 12 / median cloud_chunk_s is the bound with a parallel device (the convention behind the 9.8 / 15.8 fps of facts.md §12); the side-stream 832x480 decode of chunk c is waited for at the start of chunk c+1 (outside chunk_s, inside ttff_s / e2e_s / steady_fps_e2e); prepare_s = cloud prepare (image + T5 + VAE encode at 576x320)%s; peak_* = torch peaks of this clip (prepare + generation + streamed decode, weights of every loaded model included; the full decode that writes spec.mp4 is not)" % (" + refiner forward(s)" if REFINE else "", " + device prepare at 832x480" if REFINE else ""),
             clip_s=round(n / 16.0, 3), video_fps=fr, kind="res_split_baseline", arm=a.arm, set=a.set or task.split("__")[0], program=task.split("__")[-1] if "__" in task else None, scene=task.split("__")[1] if task.count("__") >= 2 else None,
             source=os.path.abspath(od), label=None, nulls=nulls, trim="ffmpeg -frames:v %d libx264 crf %d from the %d generated frames" % (nout, OT.CRF, n0))
    if n0 < nout: nulls["frames"] = "source has only %d frames (< %d)" % (n0, nout)
    json.dump(T, open(out + "/timing.json", "w"), indent=1)
    print(json.dumps(dict(task=task, out=out, resolution_out=T["resolution_out"], frames=n, frames_generated=n0, steps=T["steps"], gen_s=T["gen_s"], e2e_s=T["e2e_s"], ttff_s=T["ttff_s"], steady_fps_gen=T["steady_fps_gen"], steady_fps_cloud_only=T["steady_fps_cloud_only"], steady_fps_e2e=T["steady_fps_e2e"], peak_alloc_GiB=T["peak_alloc_GiB"], peak_reserved_GiB=T["peak_reserved_GiB"], nulls=nulls)), flush=True)
    if os.path.isfile(src): os.remove(src)   # purge the raw video after a successful conversion (as ours_to_timing --purge); loop.json stays
# ---------------- main ----------------
todo = [t for t in TASKS if a.force or not os.path.exists("%s/%s/timing.json" % (a.runs, t))]
print("[res_split_baseline] arm %s set %s area %d lo_area %d tasks %d (%d to run) out %s runs %s" % (a.arm, a.set, a.area, a.lo_area, len(TASKS), len(todo), a.out, a.runs), flush=True)
if not todo: print("[ALL_DONE] nothing to do", flush=True); sys.exit(0)
if a.warm: run(todo[0], a.seed + 1, a.out + "/%s/%s_warm" % (todo[0], a.arm), warm=True)
ok = 0; failed = []
for t in todo:
    od = a.out + "/" + t
    try:
        L = run(t, a.seed, od)
        if a.video: convert(t, L, od, a.runs + "/" + t); ok += 1
    except Exception as e:
        import traceback; traceback.print_exc(); print("[fail]", t, repr(e), flush=True); failed.append(t)
print("[batch] converted %d failed %d -> %s%s" % (ok, len(failed), a.runs, (" FAILED: " + ",".join(failed)) if failed else ""), flush=True); print("[ALL_DONE]", flush=True)
sys.exit(1 if failed else 0)
