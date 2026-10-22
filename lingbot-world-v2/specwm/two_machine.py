"""[10/15, item (c) of the a->b->c plan] Two-machine self-speculation = the 14B's 4-step denoising pipelined across a device and a cloud.
device (GPU A, FP8 14B, the realtime stack): drafts every chunk with the first nodes of the schedule (--use 0,1), shows it at once, sends the
  draft latent to the cloud and goes on with the next chunk; corrected chunks coming back are re-committed into its KV window (dense clean
  commit of that chunk's slot) when they differ from the draft by more than --thr (relative L2).
cloud (GPU B, bf16 14B): refines each draft with the remaining nodes (--use 2,3; re-noised to that node's sigma) in its own context (the
  corrected sequence) and returns it; when drafts queue up beyond --max-lag it commits the draft as is (skipped) so it never falls behind.
solo: one process, any node subset (teacher 0,1,2,3; realtime 0,1 of nodes 0,500; early exit 0,1 of nodes 0,250,500,750).
Transport: length-prefixed pickles over TCP (device connects to the cloud, optionally through a relay host); --delay-ms adds a one-way
delay at both receivers (WAN emulation; both processes read the same wall clock when they run on one host).
Outputs per role: <out>/<task>/{lat.pt, spec.mp4, timing.json}; summary <out>/summary.json.
usage: CUDA_VISIBLE_DEVICES=3 python -m specwm.two_machine --role cloud --port 31500 --out ABS/cloud --wq bf16 --use 2,3 --tasks ...
       CUDA_VISIBLE_DEVICES=6 python -m specwm.two_machine --role device --connect 127.0.0.1:31500 --out ABS/device --wq fp8 --use 0,1 --tasks ...
[10/19, additive, default off] --long-clip 1 (60-s P6 clips: KV caches dropped before prepare, frame-by-frame spec.mp4 decode to the host) and
--ee-head HEAD [--ee-eps 0.24 --ee-bucket 128] (device/solo drafts by token-wise early exit = the sf24 arm, baselines/run_ours.sh sf24c1hyb).
[10/19 EXP 2 light device, additive, default off] device/solo: --kv fp8 (FP8 KV ring buffer, = fit24 --kv fp8), --cap GiB (allocator cap, OOM = does not
fit, = fit24 --cap), --lowmem 1 (block-wise FP8 load: the bf16 DiT never sits on the GPU; old KV dropped before prepare; spec.mp4 decoded frame by
frame to the host with the KV resident); any of the three adds memory records (timing.json "mem", <out>/oom.json on an OOM). Cloud role unchanged.
[10/19 PM, additive, default off] device/solo: --vae-dtype bf16 / --vae-tiles R,C = spec.mp4 decoded by a bf16 copy of the VAE (weights + activations,
no autocast; prepare's encode keeps the fp32 VAE), frame by frame, optionally in R x C spatial tiles (overlap 8 latents, feathered, blended on the host)."""
import argparse, os, sys, json, time, gc, socket, struct, pickle, threading, queue, io, torch
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--role", required=True, choices=["device", "cloud", "solo"]); ap.add_argument("--out", required=True)
ap.add_argument("--tasks", default="easy_lake"); ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--frames", type=int, default=161)
ap.add_argument("--area", type=int, default=270336, help="832x480=399360, 704x384=270336"); ap.add_argument("--nodes", default="0,250,500,750"); ap.add_argument("--use", default="0,1", help="node indices this role runs")
ap.add_argument("--wq", default="fp8", choices=["bf16", "fp8", "gptq4"]); ap.add_argument("--window", type=int, default=18); ap.add_argument("--skip-blocks", default="", help="[10/16 pruning pilot] comma list of DiT block indices to bypass (identity, no training)"); ap.add_argument("--compile", type=int, default=1); ap.add_argument("--commit", default="pc1", choices=["pc1", "separate"])
ap.add_argument("--port", type=int, default=31500); ap.add_argument("--connect", default="127.0.0.1:31500"); ap.add_argument("--delay-ms", type=float, default=0.0, help="injected one-way network delay")
ap.add_argument("--thr", type=float, default=0.0, help="device re-commits a corrected chunk only if |x_corr - x_draft| / |x_draft| > thr (1e9 = never)"); ap.add_argument("--max-lag", type=int, default=1, help="cloud skips refinement when more than this many drafts are queued")
ap.add_argument("--draft-area", type=int, default=0, help="device only: draft at this (lower) resolution; the cloud upsamples the latent to its own grid before re-noising"); ap.add_argument("--draft", default="model", choices=["model", "prev", "noise"], help="device only: model = denoise with USE nodes; prev = previous chunk's latent, no compute; noise = pure noise")
ap.add_argument("--dump-feat", type=int, default=0, help="device only: save the last block's hidden after every draft step (per-channel standardized with chan_norm_14b.pt, projected 5120->1536 with gap_dump_v3/proj.pt, 2x2 pooled, fp16) to <out>/<task>/feat.pt = {h: [N, K, F, h, w, 1536]} for the draft-verifier head")
ap.add_argument("--gate", default="none", choices=["none", "head", "cos", "random", "drift", "adapt"], help="device only: which chunks are sent for cloud correction (the rest are accepted as drafted; the cloud commits them as is). head = draft-verifier head on the last draft step's features (scripts/draft_head.py, fold model of the task's held-out fold), cos = 1 - latent cosine chunk c vs c-1 (open-loop), random; chunks 0-2 always corrected")
ap.add_argument("--drift-head", default=S + "/lingbot-world-v2/specwm_out/drift_head/head_L20_all.pt"); ap.add_argument("--drift-tau", type=float, default=-3.25, help="drift gate: chunk logit > tau -> send (10/10 calibration); forces a dense commit pass (--commit separate) whose layer-20 hidden the head reads")
ap.add_argument("--gate-budget", type=float, default=0.5, help="fraction of chunks (3..) sent for correction; thresholds come from <gate-dir>/result.json (random: probability)"); ap.add_argument("--gate-dir", default="", help="draft_head.py output dir (head_fold*.pt, result.json)")
ap.add_argument("--student", default="", help="[10/16 stage 3] depth-pruned student ckpt (lora14_train --mode kd): bypassed blocks + LoRA always on"); ap.add_argument("--corr", default="", help="[10/16 stage 2] corrector LoRA ckpt (lora14_train --mode corr): active only at node 750 of this role's denoise"); ap.add_argument("--refiner", default="", help="[10/16] trained device refiner (specwm/refiner_train.py ckpt): device/solo = the SHOWN stream is xd_up + net(xd_up, eps) on the cloud grid (--area) with the cloud's own re-noise draw replicated from the seed; cloud = accepted/skipped drafts are committed refined"); ap.add_argument("--dump-pairs", type=int, default=0, help="cloud only: save per chunk the (upsampled) draft, the re-noise draw and the refined latent to <out>/<task>/pairs.pt (fp16) = training pairs for a device refiner"); ap.add_argument("--wait-s", type=float, default=20.0, help="device waits this long after the last chunk for late corrections"); ap.add_argument("--video", type=int, default=1)
ap.add_argument("--adapt-budget", type=float, default=0.0, help="[10/16 step-1 verifier, A] device/solo: after the FIRST draft step the draft-verifier head (--gate-dir, head_step1 threshold at this budget = fraction of chunks 3.. extended) scores the chunk; above threshold the chunk gets one extra denoise node (--adapt-node) = chunk-wise 2/3-step draft"); ap.add_argument("--adapt-node", type=int, default=2, help="index into --nodes of the extra step (nodes 0,500,750 -> 2 = sigma 0.625)")
ap.add_argument("--send-step", type=int, default=0, help="[10/16 step-1 verifier, B] device: send the x0 estimate after this draft step (1 = after the first step) instead of the finished draft; the cloud re-noises it to its first --use node; the device keeps drafting and shows its own finished chunk")
ap.add_argument("--timed-decode", type=int, default=0, help="[10/17 baselines, solo only, default off = unchanged] 1: decode every committed chunk right after its commit on a side CUDA stream with a second VAE instance (as spec_early --stream-decode) and record per-frame ready times + per-chunk decode seconds in timing.json (frame_ready, chunk_decode, decode_s, prep_s, peak_alloc_GiB, peak_reserved_GiB); spec.mp4 is still written by the usual full decode")
ap.add_argument("--long-clip", type=int, default=0, help="[10/19 EXP 1, default off = unchanged] 1 = long clips (P6: 969 poses): the previous clip's KV caches are dropped BEFORE prepare (prepare + reset_caches rebuild them anyway; at 1280x704 this frees one ~35 GB cache while the 969-frame conditioning is VAE-encoded) and spec.mp4 is decoded latent frame by latent frame to the host (binding_v2.decode_and_save_stream: the same frames as decode_and_save without its ~31 GB of GPU temporaries at 1280x704)")
ap.add_argument("--ee-head", default="", help="[10/19 EXP 3, default off] device/solo: token-wise early exit = Speculative Forcing on the 14B with the learned gap head (specwm.early_exit.EarlyExit; the sf24 arm of baselines/run_ours.sh: head14b_nob43/head.pt, eps 0.24, bucket 128, 9x9 + temporal smoothing) replaces the node loop of every chunk; needs --nodes 0,250,500,750 --use 0,1,2,3; blocks compiled with dynamic=True (active-token counts vary, as spec_early); per-chunk RNG gen(seed, c) = spec_early's; use with --commit separate (spec_early's dense clean commit)")
ap.add_argument("--ee-eps", type=float, default=0.24, help="[10/19] --ee-head stop threshold on the predicted token gap"); ap.add_argument("--ee-bucket", type=int, default=128, help="[10/19] --ee-head active-token padding")
ap.add_argument("--kv", default="bf16", choices=["bf16", "fp8"], help="[10/19 EXP 2 light device, default bf16 = unchanged] device/solo: dtype of the self-attention KV ring buffer; fp8 = allocated as float8_e4m3fn, written with an implicit cast and read back through fast_patch._sdpa's half() (= fit24 --kv fp8)")
ap.add_argument("--cap", type=float, default=0.0, help="[10/19 light device, default 0 = no cap] device/solo: GiB; torch.cuda.set_per_process_memory_fraction(cap / total) (= fit24 --cap): an allocation above it raises OutOfMemoryError = does not fit (written to <out>/oom.json). With --lowmem 1 the cap holds from process start (model load included), else from after the FP8 weight swap (fit24's convention)")
ap.add_argument("--vae-dtype", default="fp32", choices=["fp32", "bf16"], help="[10/19 PM light device, default fp32 = unchanged] device/solo: bf16 = spec.mp4 is decoded by a bf16 copy of the Wan VAE (bf16 weights AND activations, no autocast; ~half the decoder's activation memory; prepare's VAE encode keeps the fp32 model), latent frame by latent frame to the host; vs the fp32 decode of the same latents ~51-53 dB PSNR (10/19, PROTOCOL §7)")
ap.add_argument("--vae-tiles", default="1,1", help="[10/19 PM light device, default 1,1 = unchanged] device/solo: R,C = decode spec.mp4 in R x C spatial tiles (overlap 8 latents per interior seam, linear feathering, blended on the host in fp32); approximate (the VAE's spatial attention sees one tile); uses the --vae-dtype decode path")
ap.add_argument("--lowmem", type=int, default=0, help="[10/19 light device, default 0 = unchanged] device/solo, needs --wq fp8: 1 = (a) the bf16 DiT stays on the CPU and is FP8-quantized block by block on the GPU (the same FP8Linear tensors as the default whole-model swap; the 34.5-GiB bf16 checkpoint never sits on the GPU), (b) the previous clip's KV caches are dropped BEFORE prepare (prepare otherwise builds the new cache while the old one is alive = 2x KV), (c) spec.mp4 decoded latent frame by latent frame to the host (binding_v2.decode_and_save_stream, as --long-clip; the KV window stays resident during the decode, as on a streaming device)")
ap.add_argument("--wq-file", default="", help="[10/19 thesis, default empty = unchanged] device/solo with --wq gptq4 --lowmem 1: the offline GPTQ-4 checkpoint written by specwm.gptq_save (packed int4 group-128 weights + scales of every block linear); the DiT blocks are built on the CPU, each block's linears are replaced by packed linears from this file and the block is moved to the GPU one at a time (the bf16 weights never sit on the GPU); dequant kernel = --int4-kernel")
ap.add_argument("--int4-kernel", default="auto", choices=["auto", "triton", "torch"], help="[10/19 thesis] --wq gptq4 dequant: auto = int4_kernels.PackedInt4LinearT (fused Triton dequant, fit24 --int4-kernel triton) when specwm.int4_kernels imports, else the torch stack/reshape chain (int4_packed.PackedInt4LinearTorch); recorded in timing.json mem.int4_kernel")
ap.add_argument("--adapt-thr", type=float, default=None, help="[10/19 audit] device/solo, with --adapt-budget > 0: override the step-1 head threshold read from <gate-dir>/result.json (thr.head_step1[budget]) with this value (baselines/adapt_thr.json by set: run_ours.sh arms adapt25r / adapt25rc1hyb); default None = the result.json threshold, unchanged")
a = ap.parse_args(); a.out = os.path.abspath(a.out)
_NEW_DEFAULTS = dict(long_clip=0, ee_head="", ee_eps=0.24, ee_bucket=128, kv="bf16", cap=0.0, lowmem=0, vae_dtype="fp32", vae_tiles="1,1", adapt_thr=None, wq_file="", int4_kernel="auto")   # [10/19 thesis] + wq_file / int4_kernel   # [10/19] omitted from summary.json while at their defaults (default path: identical summary)
assert a.role != "cloud" or (a.kv == "bf16" and a.cap == 0 and not a.lowmem), "[10/19] --kv / --cap / --lowmem are device/solo options (the cloud is unchanged)"
assert not a.lowmem or a.wq in ("fp8", "gptq4"), "[10/19] --lowmem 1 needs --wq fp8 (it quantizes the blocks while loading) or --wq gptq4 --wq-file F (it loads the packed blocks)"
assert a.wq != "gptq4" or (a.role != "cloud" and a.lowmem and a.wq_file and os.path.isfile(a.wq_file)), "[10/19 thesis] --wq gptq4 is a device/solo option and needs --lowmem 1 --wq-file <existing specwm.gptq_save file>"   # the cloud is unchanged
VTILES = tuple(int(v) for v in a.vae_tiles.split(",")); assert len(VTILES) == 2 and min(VTILES) >= 1, "--vae-tiles R,C"
LIGHTDEC = a.vae_dtype != "fp32" or VTILES != (1, 1)   # [10/19 PM] the light-device decode path (default: off)
assert a.role != "cloud" or not LIGHTDEC, "[10/19] --vae-dtype / --vae-tiles are device/solo options (the cloud is unchanged)"
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
GiB = 2 ** 30; dev = "cuda:0"; sync = torch.cuda.synchronize; os.makedirs(a.out, exist_ok=True); now = time.time
def log(*m): print("[%s %s]" % (a.role, time.strftime("%H:%M:%S")), *m, flush=True)
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}   # [10/16 baselines] SPECWM_TASKS=extra.json adds/overrides task entries (same schema: id, image, camera_WM, action dir with poses.npy + intrinsics.npy)
if os.environ.get("SPECWM_TASKS"): ALL.update({t["id"]: t for t in json.load(open(os.environ["SPECWM_TASKS"]))})
FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
NODES = tuple(int(v) for v in a.nodes.split(",")); USE = [int(v) for v in a.use.split(",")]
# ---- [10/19 EXP 2 light device] memory records (only when --kv fp8 / --cap / --lowmem is given; the default path never runs any of this) ----
MEMREC = bool(a.lowmem or a.cap > 0 or a.kv != "bf16" or LIGHTDEC or a.wq == "gptq4"); MEM = {}; STAGE = ["load"]; NVSMI = {"peak": None, "last": None}
def _mem_mark(tag):
    """torch peak allocated / reserved since the last reset_peak_memory_stats + allocated now (GiB), and this process's nvidia-smi used memory (MiB)"""
    sync(); MEM[tag + "_peak_alloc_GiB"] = round(torch.cuda.max_memory_allocated() / GiB, 3); MEM[tag + "_peak_reserved_GiB"] = round(torch.cuda.max_memory_reserved() / GiB, 3); MEM[tag + "_alloc_GiB"] = round(torch.cuda.memory_allocated() / GiB, 3)
    MEM["proc_peak_alloc_GiB"] = max(MEM.get("proc_peak_alloc_GiB", 0.0), MEM[tag + "_peak_alloc_GiB"]); MEM["proc_peak_reserved_GiB"] = max(MEM.get("proc_peak_reserved_GiB", 0.0), MEM[tag + "_peak_reserved_GiB"])
    MEM[tag + "_nvsmi_MiB"] = NVSMI["last"]; MEM["nvsmi_proc_peak_MiB"] = NVSMI["peak"]
    log("[mem] %-11s peak alloc %.2f / reserved %.2f GiB, alloc now %.2f GiB, nvidia-smi now %s MiB (process peak so far %s MiB)" % (tag, MEM[tag + "_peak_alloc_GiB"], MEM[tag + "_peak_reserved_GiB"], MEM[tag + "_alloc_GiB"], NVSMI["last"], NVSMI["peak"]))
def _apply_cap(where):
    if a.cap > 0:
        gc.collect(); torch.cuda.empty_cache(); tot = torch.cuda.get_device_properties(0).total_memory / GiB; torch.cuda.set_per_process_memory_fraction(min(1.0, a.cap / tot))
        MEM.update(cap_GiB=a.cap, cap_applied=where, gpu_total_GiB=round(tot, 2)); log("cap: CUDA caching allocator limited to %.1f GiB of %.1f GiB (%s)" % (a.cap, tot, where))
if MEMREC:
    def _nvsmi_poll():   # this process's used GPU memory as nvidia-smi reports it (CUDA context + every allocation), polled every 0.5 s (sub-second transients can be missed; torch's reserved peak is exact)
        import subprocess as _sp
        me = str(os.getpid())
        while True:
            try:
                for ln in _sp.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout.splitlines():
                    p_ = [x.strip() for x in ln.split(",")]
                    if len(p_) >= 2 and p_[0] == me and p_[1].isdigit(): NVSMI["last"] = int(p_[1]); NVSMI["peak"] = max(NVSMI["peak"] or 0, int(p_[1]))
            except Exception: pass
            time.sleep(0.5)
    threading.Thread(target=_nvsmi_poll, daemon=True).start()
    _EH0 = sys.excepthook
    def _oom_hook(et, ev, tb):   # an uncaught OOM under --cap = the configuration does not fit: say so and keep the numbers
        if issubclass(et, torch.OutOfMemoryError):
            msg = str(ev).split("\n")[0][:300]
            try: json.dump(dict(oom_stage=STAGE[0], role=a.role, cap_GiB=a.cap, kv=a.kv, window=a.window, area=a.area, wq=a.wq, lowmem=a.lowmem, tasks=a.tasks, msg=msg, mem=MEM, nvsmi_proc_peak_MiB=NVSMI["peak"],
                                peak_alloc_GiB_since_reset=round(torch.cuda.max_memory_allocated() / GiB, 3), peak_reserved_GiB_since_reset=round(torch.cuda.max_memory_reserved() / GiB, 3), t=time.strftime("%F %T")), open(a.out + "/oom.json", "w"), indent=1)
            except Exception: pass
            print("[%s] OOM at stage %s (cap %.1f GiB, kv %s, window %d, area %d): DOES NOT FIT -- %s" % (a.role, STAGE[0], a.cap, a.kv, a.window, a.area, msg), flush=True)
        _EH0(et, ev, tb)
    sys.excepthook = _oom_hook
    MEM.update(kv=a.kv, window=a.window, wq=a.wq, lowmem=a.lowmem, cap_GiB=a.cap, area=a.area, vae_dtype=a.vae_dtype, vae_tiles=a.vae_tiles)
if a.lowmem:   # (a) the pipe's own .to(device) of the bf16 DiT (right after _configure_model) is skipped once: the model stays on the CPU; V2Binding(swap_models=True) leaves it there
    _cap_where = "process start (before the model load)"; _apply_cap(_cap_where)
    import wan.image2video as _WI
    _CFG0 = _WI.WanI2VCausal._configure_model
    def _cfg_cpu(self_, *x_, **k_):
        m_ = _CFG0(self_, *x_, **k_)
        def _skip_to(*x2, **k2):
            del m_.to; return m_
        m_.to = _skip_to; return m_
    _WI.WanI2VCausal._configure_model = _cfg_cpu
Tg = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=a.window, swap_models=bool(a.lowmem), build_draft=False, timesteps_index=NODES); tm = Tg.target_model
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True)
if a.lowmem:   # (a) block by block: bf16 block -> GPU -> FP8Linear (fast_patch.fp8_swap on that block alone, computed on the GPU exactly as the whole-model swap) -> bf16 copy freed
    import types as _ty
    _WI.WanI2VCausal._configure_model = _CFG0; assert next(tm.parameters()).device.type == "cpu", "--lowmem: the DiT should still be on the CPU here"; _t0 = now()
    if a.wq == "gptq4":   # [10/19 thesis] packed INT4 blocks from the offline GPTQ-4 file: linears replaced on the CPU, then the block moves (never the bf16 weights)
        from specwm.int4_packed import load_block as _ld4
        _Q = torch.load(a.wq_file, map_location="cpu", weights_only=True); _QL = _Q["layers"]; _nq = 0; _miss = []; _kern = None
        for _bi, _blk in enumerate(tm.blocks):
            _n4, _kern, _m4 = _ld4(_blk, "blocks.%d" % _bi, _QL, a.int4_kernel); _nq += _n4; _miss += _m4; _blk.to(dev)
        tm.to(dev); Tg.swap_models = False; Tg._resident = "target"; _qm = _Q.get("meta", {}); del _Q, _QL; gc.collect(); torch.cuda.empty_cache()
        MEM.update(wq_file=a.wq_file, wq_layers=_nq, int4_kernel=_kern, wq_meta={k: _qm.get(k) for k in ("calib_tasks", "resolution", "nodes", "window", "seed", "damp", "redamped", "nonfinite", "made", "group", "bits")}, compile=a.compile)
        log("lowmem: DiT loaded on the CPU, %d block linears replaced by packed INT4 (GPTQ-4 g%s, %s; dequant kernel %s) from %s in %.0f s; %d large linears without an entry %s; allocated %.2f GiB, peak %.2f GiB"
            % (_nq, _qm.get("group"), _qm.get("calib_tasks"), _kern, a.wq_file, now() - _t0, len(_miss), _miss[:3], torch.cuda.memory_allocated() / GiB, torch.cuda.max_memory_allocated() / GiB))
        assert _nq > 0 and not _miss, "[10/19 thesis] --wq-file: every block linear with in_features >= 1024 must have an entry (missing %s)" % _miss[:5]
    else:
        for _blk in tm.blocks: _blk.to(dev); FP.fp8_swap(_ty.SimpleNamespace(blocks=_blk), True, keep=False)
        tm.to(dev); Tg.swap_models = False; Tg._resident = "target"; gc.collect(); torch.cuda.empty_cache()
        log("lowmem: DiT loaded on the CPU, FP8-quantized block by block on the GPU in %.0f s; allocated %.2f GiB, peak %.2f GiB" % (now() - _t0, torch.cuda.memory_allocated() / GiB, torch.cuda.max_memory_allocated() / GiB))
if a.wq == "fp8": FP.fp8_swap(tm, True, keep=False)
CGATE = None
if a.student or a.corr:
    from specwm import lora_util as LU; _ck = torch.load(a.student or a.corr, map_location="cpu")
    if a.corr: CGATE = torch.zeros((), device=dev)
    _lor, _g = LU.load(tm, _ck, dev, gate=CGATE); log("lora", a.student or a.corr, "mode", _ck.get("mode"), "rank", _ck["rank"], "skip", _ck.get("skip"), "step", _ck.get("step"), "params %.0fM" % (sum(l.A.numel() + l.B.numel() for l in _lor) / 1e6))
if a.skip_blocks:
    SKIP = [int(v) for v in a.skip_blocks.split(",")]
    for i in SKIP: tm.blocks[i].forward = (lambda x, **kw: x)
    log("skip-blocks (identity, untrained):", SKIP, "=> %d of %d blocks active" % (len(tm.blocks) - len(SKIP), len(tm.blocks)))
gc.collect(); torch.cuda.empty_cache()
if a.cap > 0 and not a.lowmem: _apply_cap("after the FP8 weight swap (the bf16 checkpoint was loaded to the GPU first)")
if a.kv == "fp8":   # [10/19 light device] = fit24 --kv fp8: only the allocation dtype of the self-attention ring buffer changes
    _init_kv = Tg.pipe._initialize_self_kv_cache
    Tg.pipe._initialize_self_kv_cache = lambda num_layers, shape, dtype, device: _init_kv(num_layers=num_layers, shape=shape, dtype=torch.float8_e4m3fn, device=device)
    log("kv: self-attention KV ring buffer allocated as float8_e4m3fn")
VAEW = None
if LIGHTDEC:   # [10/19 PM] the decode model: a bf16 copy of the VAE (--vae-dtype bf16) or the fp32 VAE itself (tiles only)
    import copy as _cp
    VAEW = _cp.deepcopy(Tg.pipe.vae.model).to(torch.bfloat16) if a.vae_dtype == "bf16" else Tg.pipe.vae.model; VAEW.clear_cache()
    log("vae decode: %s weights + activations (no autocast), tiles %dx%d, frame by frame to the host" % (a.vae_dtype, VTILES[0], VTILES[1]))
@torch.no_grad()
def _decode_light(lat, ov=8, sc=8):
    """[10/19 PM] spec.mp4 frames (uint8 [T, H, W, 3], decode_and_save's conversion) from VAEW in its own dtype, latent frame by latent frame, in VTILES spatial
    tiles (overlap ov latents per interior seam, linear feathering, blended on the host); the scale/shift of the latent is applied in fp32 first"""
    V = Tg.pipe.vae; m = VAEW; dt = next(m.parameters()).dtype; z = lat.to(dev).float().unsqueeze(0); z = z / V.scale[1].view(1, -1, 1, 1, 1) + V.scale[0].view(1, -1, 1, 1, 1)
    Hl, Wl = z.shape[-2:]; R, C = VTILES; hs = [(Hl * i) // R for i in range(R)] + [Hl]; ws = [(Wl * j) // C for j in range(C)] + [Wl]; acc = wsum = None
    for i in range(R):
        for j in range(C):
            h0, h1, w0, w1 = hs[i], hs[i + 1], ws[j], ws[j + 1]; ph0, ph1, pw0, pw1 = max(0, h0 - ov), min(Hl, h1 + ov), max(0, w0 - ov), min(Wl, w1 + ov)
            zt = z[..., ph0:ph1, pw0:pw1].to(dt); m.clear_cache(); x = m.conv2(zt); fr = []
            for f in range(zt.shape[2]):
                m._conv_idx = [0]; fr.append(m.decoder(x[:, :, f:f + 1], feat_cache=m._feat_map, feat_idx=m._conv_idx).float().clamp_(-1, 1).squeeze(0).cpu())
            m.clear_cache(); del x, zt; piece = torch.cat(fr, 1); del fr
            if R * C == 1: return ((piece.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 3, 0).numpy()
            if acc is None: acc = torch.zeros(piece.shape[0], piece.shape[1], Hl * sc, Wl * sc); wsum = torch.zeros(1, 1, Hl * sc, Wl * sc)
            ph, pw = piece.shape[-2:]; mk = torch.ones(1, 1, ph, pw); top, bot, lef, rig = (h0 - ph0) * sc, (ph1 - h1) * sc, (w0 - pw0) * sc, (pw1 - w1) * sc
            if top: mk[..., :top, :] *= torch.linspace(0, 1, top).view(1, 1, -1, 1)
            if bot: mk[..., ph - bot:, :] *= torch.linspace(1, 0, bot).view(1, 1, -1, 1)
            if lef: mk[..., :, :lef] *= torch.linspace(0, 1, lef).view(1, 1, 1, -1)
            if rig: mk[..., :, pw - rig:] *= torch.linspace(1, 0, rig).view(1, 1, 1, -1)
            acc[:, :, ph0 * sc:ph0 * sc + ph, pw0 * sc:pw0 * sc + pw] += piece * mk[0]; wsum[:, :, ph0 * sc:ph0 * sc + ph, pw0 * sc:pw0 * sc + pw] += mk[0]; del piece
    o = acc / wsum.clamp(min=1e-6); return ((o.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 3, 0).numpy()
def _decode_light_save(lat, od):
    import imageio
    torch.cuda.empty_cache()   # cached blocks of the generation's activations go back first (the KV window stays allocated)
    if MEMREC: torch.cuda.reset_peak_memory_stats()
    v = _decode_light(lat); imageio.mimwrite(os.path.join(od, "spec.mp4"), list(v), fps=16); return len(v)
if MEMREC: STAGE[0] = "warmup"; _mem_mark("load"); torch.cuda.reset_peak_memory_stats()
if a.compile:
    import torch._dynamo as _dyn; _dyn.config.recompile_limit = 512; _dyn.config.accumulated_recompile_limit = 8192
    for blk in tm.blocks: blk.forward = torch.compile(blk.forward, dynamic=bool(a.ee_head))   # [10/19] --ee-head: dynamic shapes like spec_early (default: static, unchanged)
enc = Tg.pipe.text_encoder; _cache = {}
class _C:
    def __init__(s): s.model = enc.model
    def __call__(s, texts, device):
        k = (tuple(texts), str(device))
        if k not in _cache: _cache[k] = enc(texts, device)
        return _cache[k]
Tg.pipe.text_encoder = _C(); SIG = [float(s) for s in Tg.node_sigmas()]; log("nodes", NODES, "sigmas", [round(s, 3) for s in SIG], "use", USE, "wq", a.wq, "window", a.window)
def gen(seed, c, k=0): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k * 100003)
CS = None; HH = {}; FEATS = []; REC = None; PAIRS = []
if a.dump_feat or a.gate == "head" or a.adapt_budget > 0:
    PROJ = torch.load(S + "/lingbot-world-v2/specwm_out/gap_dump_v3/proj.pt").to(dev, torch.bfloat16); _n = torch.load(S + "/lingbot-world-v2/specwm_out/chan_norm_14b.pt"); MU, SD = _n["mu"].to(dev), _n["sd"].to(dev)
    def _hook(m, inp, out): HH["h"] = out[0] if isinstance(out, tuple) else out
    tm.blocks[-1].register_forward_hook(_hook)
    @torch.no_grad()
    def feat_now():
        f = (((HH["h"][0].float() - MU) / SD).to(torch.bfloat16) @ PROJ); F_, h2, w2 = Tg._chunk_size, Tg._chunk_shape[2] // 2, Tg._chunk_shape[3] // 2
        return torch.nn.functional.avg_pool2d(f.view(F_, h2, w2, -1).permute(0, 3, 1, 2).float(), 2, ceil_mode=True).permute(0, 2, 3, 1).half().cpu()
@torch.no_grad()
def denoise(c, g, start_x=None, feats=None, eps0=None, on_step=None, adapt=False):
    """run the nodes in USE for chunk c; from noise (USE[0] must be node 0) or from a draft x0 re-noised to the first used node (eps0 = that draw, if given).
    on_step(j, x) is called after every step (early send); adapt = the step-1 head may append ADAPT['node'] to this chunk's node list"""
    x = start_x; use = list(USE); j = 0
    while j < len(use):
        i = use[j]
        if j == 0 and start_x is None: z = Tg.chunk_noise(c, g)
        else:
            s_ = SIG[i]; eps = eps0 if (j == 0 and eps0 is not None) else Tg.randn_like(x, g); z = (1 - s_) * x + s_ * eps
            if REC is not None: REC.append(eps)
        if CGATE is not None: CGATE.fill_(1.0 if NODES[i] == 750 else 0.0)
        x = Tg.target_x0(z, i, c).float()
        if CGATE is not None: CGATE.fill_(0.0)
        if feats is not None: feats.append(feat_now())
        if on_step is not None: on_step(j, x)
        if adapt and j == 0 and ADAPT is not None:
            fi = next(k for k, f in enumerate(ADAPT["folds"]) if CUR["task"] in f); sc = float(ADAPT["heads"][fi](feats[-1][None].to(dev)).mean()); ext = bool(c < 3 or sc > ADAPT["thr"])
            if ext: use.append(ADAPT["node"])
            ADAPT["log"].append(dict(c=c, sc=round(sc, 4), ext=ext))
        j += 1
    return x
ADAPT = None; CUR = {"task": None}
if a.adapt_budget > 0:
    import torch.nn as _nn
    class AHead(_nn.Module):   # = scripts/draft_head.py Head (trained on the features of every draft step; result.json has its step-1 thresholds)
        def __init__(s, cin=1536):
            super().__init__(); s.ln = _nn.LayerNorm(cin); s.c1 = _nn.Conv3d(cin, 96, 3, padding=1); s.c2 = _nn.Conv3d(96, 48, 3, padding=1); s.c3 = _nn.Conv3d(48, 1, 1)
        def forward(s, h):
            x = s.ln(h.float()).permute(0, 4, 1, 2, 3); x = torch.nn.functional.gelu(s.c1(x)); x = torch.nn.functional.gelu(s.c2(x)); return s.c3(x).squeeze(1)
    _GR = json.load(open(a.gate_dir + "/result.json")); ADAPT = dict(thr=_GR["thr"]["head_step1"]["%.2f" % a.adapt_budget], folds=_GR["folds"], node=a.adapt_node, heads=[], log=[])
    ADAPT["thr_source"] = "%s/result.json thr.head_step1[%.2f]" % (a.gate_dir, a.adapt_budget)   # [10/19 audit] recorded in timing.json (adapt_thr_source)
    if a.adapt_thr is not None: ADAPT["thr"] = float(a.adapt_thr); ADAPT["thr_source"] = "--adapt-thr %r" % a.adapt_thr   # [10/19 audit] per-set re-calibrated threshold (baselines/adapt_thr.json); default None = unchanged
    for fi in range(len(_GR["folds"])):
        hm = AHead().to(dev); hm.load_state_dict(torch.load("%s/head_fold%d.pt" % (a.gate_dir, fi), map_location=dev)); hm.eval(); ADAPT["heads"].append(hm)
    assert a.adapt_node < len(NODES) and a.adapt_node not in USE, (a.adapt_node, NODES, USE); log("adapt: step-1 head", a.gate_dir, "budget", a.adapt_budget, "thr", ADAPT["thr"], "extra node", a.adapt_node, "sigma", round(SIG[a.adapt_node], 3), "thr source", ADAPT["thr_source"])
assert not (a.send_step and a.gate != "none"), "early send is always-send (gate scores need the finished draft)"
GATE = {}
if a.gate != "none":
    import torch.nn as nn, random as _rnd
    GATE["rng"] = _rnd.Random(a.seed); GATE["B"] = a.gate_budget
    if a.gate in ("head", "cos"):
        GR = json.load(open(a.gate_dir + "/result.json")); key = "head_step%d" % len(USE) if a.gate == "head" else "latent_cos_1-cos"; GATE["thr"] = GR["thr"][key]["%.2f" % a.gate_budget]; GATE["folds"] = GR["folds"]
    if a.gate == "head":
        class Head(nn.Module):
            def __init__(s, cin=1536):
                super().__init__(); s.ln = nn.LayerNorm(cin); s.c1 = nn.Conv3d(cin, 96, 3, padding=1); s.c2 = nn.Conv3d(96, 48, 3, padding=1); s.c3 = nn.Conv3d(48, 1, 1)
            def forward(s, h):
                x = s.ln(h.float()).permute(0, 4, 1, 2, 3); x = torch.nn.functional.gelu(s.c1(x)); x = torch.nn.functional.gelu(s.c2(x)); return s.c3(x).squeeze(1)
        GATE["heads"] = []
        for fi in range(len(GATE["folds"])):
            hm = Head().to(dev); hm.load_state_dict(torch.load("%s/head_fold%d.pt" % (a.gate_dir, fi), map_location=dev)); hm.eval(); GATE["heads"].append(hm)
    if a.gate == "drift":   # 10/9 scene-general drift head: MLP over 16 pooled tokens (global mean + 3x5) of the layer-20 hidden of the dense commit pass
        class DHead(nn.Module):
            def __init__(s, C, hid):
                super().__init__(); s.mu = nn.Parameter(torch.zeros(C), requires_grad=False); s.sd = nn.Parameter(torch.ones(C), requires_grad=False)
                s.tok = nn.Sequential(nn.Linear(C, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU()); s.pos = nn.Parameter(torch.zeros(16, hid)); s.out = nn.Sequential(nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))
            def chunk_logits(s, x): h = s.tok((x - s.mu) / s.sd) + s.pos; return s.out(h.mean(1)).squeeze(-1)
        hk = torch.load(a.drift_head, map_location="cpu"); dh = DHead(5120, hk["hid"]).to(dev); dh.load_state_dict(hk["state"]); dh.eval(); GATE["dhead"] = dh; GATE["thr"] = a.drift_tau; a.commit = "separate"
        def _h20(m, i, o): HH["h20"] = (o[0] if isinstance(o, tuple) else o)
        tm.blocks[19].register_forward_hook(_h20)
    log("gate", a.gate, "budget", a.gate_budget, "thr", GATE.get("thr"), "commit", a.commit)
@torch.no_grad()
def gate_score(task, c, x, feats, prev_x):
    """returns (score, refine) for chunk c of task; chunks 0-2 are always refined"""
    if a.gate == "adapt":   # [10/16 A+B] send only the chunks the step-1 head extended (--adapt-budget): the cloud corrects what the device itself flagged
        e = ADAPT["log"][-1]; return e["sc"], e["ext"]
    if a.gate == "head":
        fi = next(i for i, f in enumerate(GATE["folds"]) if task in f); sc = float(GATE["heads"][fi](feats[-1][None].to(dev)).mean())
    elif a.gate == "cos": sc = 1.0 - float(torch.nn.functional.cosine_similarity(x.flatten()[None], prev_x.flatten()[None])) if prev_x is not None else 1.0
    elif a.gate == "drift":   # HH["h20"] = layer-20 output of the dense commit pass just run for this chunk
        f_, h2, w2 = Tg._chunk_size, Tg._chunk_shape[2] // 2, Tg._chunk_shape[3] // 2; hx = HH["h20"][0].float().view(f_, h2, w2, -1).permute(3, 0, 1, 2)
        g = torch.nn.functional.adaptive_avg_pool3d(hx, (1, 3, 5)).reshape(hx.shape[0], 15).t(); sc = float(GATE["dhead"].chunk_logits(torch.cat([hx.mean((1, 2, 3))[None], g])[None])[0])
    else: sc = GATE["rng"].random()
    thr = GATE["thr"] if a.gate != "random" else 1.0 - GATE["B"]
    return sc, bool(c < 3 or sc > thr)
EE = None; EE_LOG = []
if a.ee_head:   # [10/19 EXP 3] token-wise early exit on this role's drafts (= the sf24 arm's settings)
    assert a.role in ("device", "solo") and NODES == (0, 250, 500, 750) and USE == [0, 1, 2, 3], "--ee-head needs --role device|solo --nodes 0,250,500,750 --use 0,1,2,3"
    assert not (a.adapt_budget > 0 or a.gate != "none" or a.send_step or a.dump_feat or a.draft != "model" or a.refiner or a.draft_area or a.student or a.corr), "--ee-head is the plain sf24 draft (no adapt / gate / early send / refiner / draft-area)"
    from specwm.early_exit import EarlyExit
    EE = EarlyExit(Tg, eps=a.ee_eps, head=a.ee_head, bucket=a.ee_bucket, device=dev); log("early exit: head", a.ee_head, "eps", a.ee_eps, "bucket", a.ee_bucket, "commit", a.commit)
@torch.no_grad()
def ee_draft(c, seed=None):
    """[10/19 EXP 3] one chunk by token-wise early exit; fresh per-chunk generator gen(seed, c) = spec_early.gen(seed, c) (noise + re-noise draws)"""
    g_ = gen(a.seed if seed is None else seed, c); z = Tg.chunk_noise(c, g_); x, ms, hist = EE.gen_chunk(c, z, g_)
    EE_LOG.append(dict(c=c, mean_steps=round(ms, 4), hist=hist)); return x
@torch.no_grad()
def commit(x, c, mode=None):
    mode = mode or a.commit
    if mode == "pc1":   # fused + partial commit 1: K/V of the last denoise step kept, only the last latent frame re-written clean (fit24 / bench-43 "ours")
        fs = Tg._frame_seqlen; L = Tg._chunk_size * fs; FP.ACTIVE["a"] = dict(idx=torch.arange(L - fs, L, device=dev), n=fs, L=L)
        try: Tg.commit(x, c)
        finally: FP.ACTIVE["a"] = None
    else: Tg.commit(x, c)
@torch.no_grad()
def recommit(x, c):
    """dense clean commit of an older chunk's slot (the corrected chunk replaces the draft's K/V); ring_kv addresses the slot by chunk id"""
    Tg.commit(x, c, which="target"); Tg._committed_frames -= Tg._chunk_size
def rel(xa, xb): return float((xa - xb).norm() / xb.norm().clamp_min(1e-6))
def slots():   # local chunks the ring keeps besides the sink
    return (a.window - 6) // Tg._chunk_size
# ---- transport ----
def send(sock, obj):
    b = pickle.dumps(obj, protocol=4); sock.sendall(struct.pack("!I", len(b)) + b)
def recv_all(sock, n):
    buf = b""
    while len(buf) < n:
        p = sock.recv(n - len(buf))
        if not p: raise EOFError
        buf += p
    return buf
def recv(sock): return pickle.loads(recv_all(sock, struct.unpack("!I", recv_all(sock, 4))[0]))
def rx_thread(sock, q):
    try:
        while True: q.put(recv(sock))
    except Exception as e: q.put(("EOF", repr(e)))
def delay_until(t_send):
    if a.delay_ms > 0:
        d = t_send + a.delay_ms / 1000.0 - now()
        if d > 0: time.sleep(d)
REF = None
if a.refiner:
    from specwm.refiner_net import RefinerNet; _ck = torch.load(a.refiner, map_location="cpu"); REF = RefinerNet(c0=_ck["c0"], c1=_ck["c1"]).to(dev); REF.load_state_dict(_ck["state"]); REF.eval(); log("refiner", a.refiner, "step", _ck.get("step"), "val", _ck.get("val"))
def cloud_grid(task):
    """latent (h, w) the cloud uses for this task at --area (binding_v2.prepare formula)"""
    from PIL import Image; import numpy as np; w, h = Image.open(ALL[task]["image"]).size; asp = h / w
    return int(round(np.sqrt(a.area * asp) // 8 // 2 * 2)), int(round(np.sqrt(a.area / asp) // 8 // 2 * 2))
def up(x, hw): return x if tuple(x.shape[2:]) == tuple(hw) else torch.nn.functional.interpolate(x.permute(1, 0, 2, 3), size=tuple(hw), mode="bicubic", align_corners=False).permute(1, 0, 2, 3).contiguous()
@torch.no_grad()
def refine_local(x, c, hw, gc):
    """device-side stand-in for the cloud's one-step refine: same upsample, same eps draw (cloud generator replica gc), trained net"""
    eps = torch.randn((x.shape[0], x.shape[1]) + tuple(hw), generator=gc).to(dev); xd = up(x.half().float(), hw)
    return REF.refine(xd[None], eps[None])[0].float()
def prepare(task):
    if MEMREC and STAGE[0] != "warmup": STAGE[0] = "prepare"; torch.cuda.reset_peak_memory_stats()   # [10/19] per-clip memory stats (the first prepare belongs to the warm-up)
    if a.long_clip or a.lowmem:   # [10/19 EXP 1] drop the previous clip's KV caches before the long VAE encode (rebuilt by prepare + reset_caches below); [10/19 --lowmem (b)] the same for every clip
        import gc as _gcm; Tg._caches = {}; _gcm.collect(); torch.cuda.empty_cache()
    T = ALL[task]; cond = Tg.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=a.frames, max_area=(a.draft_area if a.role in ("device", "solo") and a.draft_area else a.area)); Tg.reset_caches()
    if MEMREC and STAGE[0] == "prepare": STAGE[0] = "generate"
    return cond["num_chunks"]
def finish_clip(task, chunks, timing):
    od = a.out + "/" + task; os.makedirs(od, exist_ok=True); lat = torch.cat(chunks, 1)
    if MEMREC: STAGE[0] = "decode"; _mem_mark("clip_gen"); torch.cuda.reset_peak_memory_stats()   # [10/19] clip_gen = prepare + generation (+ device re-commits) of this clip
    torch.save(lat.half().cpu(), od + "/lat.pt"); json.dump(timing, open(od + "/timing.json", "w"), indent=1)
    if a.video and LIGHTDEC: _decode_light_save(lat, od)   # [10/19 PM] --vae-dtype / --vae-tiles
    elif a.video: (Tg.decode_and_save_stream if (a.long_clip or a.lowmem) else Tg.decode_and_save)([lat.to(dev)], od)   # [10/19] --long-clip / --lowmem: frame-by-frame decode to the host
    if MEMREC:   # [10/19] decode peak (KV window still resident), records into timing.json (rewritten) and summary.json
        _mem_mark("clip_decode"); MEM["clip_peak_alloc_GiB"] = max(MEM["clip_gen_peak_alloc_GiB"], MEM["clip_decode_peak_alloc_GiB"]); MEM["clip_peak_reserved_GiB"] = max(MEM["clip_gen_peak_reserved_GiB"], MEM["clip_decode_peak_reserved_GiB"])
        timing["mem"] = dict(MEM); json.dump(timing, open(od + "/timing.json", "w"), indent=1); SUMM.setdefault("mem", {})[task] = dict(MEM); STAGE[0] = "between clips"
def warmup(N):
    """compile every shape on a throw-away clip: the role's denoise + commit, plus a dense commit (device re-commit / cloud skip)"""
    g = gen(a.seed + 1, 0); x = None
    for c in range(N):
        x = ee_draft(c, a.seed + 1) if EE is not None else denoise(c, g, None if USE[0] == 0 else torch.randn(Tg._chunk_shape, generator=g).to(dev)); commit(x, c)
    recommit(x, N - 1)
    if MEMREC: _mem_mark("warm")   # [10/19] first prepare + compile warm-up (all shapes incl. the dense re-commit), KV of the warm clip still allocated
    Tg.reset_caches(); gc.collect(); torch.cuda.empty_cache(); sync(); torch.cuda.reset_peak_memory_stats()
    if MEMREC: STAGE[0] = "between clips"
# ---- [10/17 --timed-decode 1, solo only] streaming decode on a side CUDA stream (copy of spec_early.StreamDecoder; frames kept on the host as uint8 only)
VAE2 = None
if a.timed_decode and a.role == "solo":
    from wan.modules.vae2_1 import Wan2_1_VAE
    VAE2 = Wan2_1_VAE(vae_pth=FAST + "/Wan2.1_VAE.pth", device=dev, dtype=Tg.pipe.vae.dtype); log("timed-decode: second VAE instance on a side stream")
class StreamDecoder:
    def __init__(s, vae): s.v = vae; s.m = vae.model; s.m.clear_cache(); s.st = torch.cuda.Stream(); s.ready = []; s.nf = 0; s.lock = threading.Lock(); s.th = []; s.ev = []; s.host = []
    def _watch(s, ev, ids, chunk):
        ev.synchronize(); t = now()
        with s.lock: s.ready += [(i, chunk, t) for i in ids]
    def push(s, z, chunk=-1):
        """z = committed chunk latent [16, F, h, w] (float); enqueues its decode on the side stream and returns at once"""
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
        """wait for every queued decode; returns per-chunk [{c, push_t, gpu_ms}] (gpu_ms = CUDA-event time of that chunk's decode on the side stream)"""
        torch.cuda.current_stream().wait_stream(s.st); sync(); [t.join() for t in s.th]; s.m.clear_cache()
        return [dict(c=c, push_t=tp, gpu_ms=round(e0.elapsed_time(e1), 2)) for c, tp, e0, e1 in s.ev]
TASKS = [t for t in a.tasks.split(",") if t]; SUMM = dict({k: v for k, v in vars(a).items() if not (k in _NEW_DEFAULTS and v == _NEW_DEFAULTS[k])}, sigmas=SIG, clips={})
# ================= roles =================
if a.role == "solo":
    N = prepare(TASKS[0]); warmup(N)
    for task in TASKS:
        tp0 = now(); N = prepare(task); SDEC = None; prep_s = None
        if VAE2 is not None: sync(); prep_s = now() - tp0; SDEC = StreamDecoder(VAE2)   # [10/17 --timed-decode] prepare = image + T5 + VAE encode, timed
        g = gen(a.seed, 0); chunks = []; tc = []; t0 = now(); gc = gen(a.seed, 0, 1); hw = cloud_grid(task) if REF is not None else None; CUR["task"] = task
        if ADAPT is not None: ADAPT["log"] = []
        EE_LOG.clear()
        for c in range(N):
            sync(); t1 = now(); fe = [] if ADAPT is not None else None; x = ee_draft(c) if EE is not None else denoise(c, g, feats=fe, adapt=ADAPT is not None); commit(x, c)
            if REF is not None: x = refine_local(x, c, hw, gc)
            sync(); tc.append(round(now() - t1, 4)); chunks.append(x)
            if SDEC is not None: SDEC.push(x.float(), c)   # [10/17 --timed-decode] decode the committed chunk on the side stream (the next chunk's sync() waits for it: sequential DiT + VAE like the baselines' e2e)
        st = sorted(tc[3:]); timing = dict(task=task, chunk_s=tc, steady_chunk_s=st[len(st) // 2], fps=round(12.0 / st[len(st) // 2], 2), clip_s=round(now() - t0, 1), peak_GiB=round(torch.cuda.max_memory_allocated() / GiB, 2),
                      mean_chunk_s=round(sum(tc[3:]) / max(1, len(tc[3:])), 4), fps_mean=round(12.0 / (sum(tc[3:]) / max(1, len(tc[3:]))), 2), adapt=(ADAPT["log"] if ADAPT is not None else None), n_ext=(sum(1 for r_ in ADAPT["log"] if r_["ext"] and r_["c"] >= 3) if ADAPT is not None else None))
        if ADAPT is not None: timing.update(adapt_thr=ADAPT["thr"], adapt_thr_source=ADAPT["thr_source"])   # [10/19 audit] the effective step-1 threshold and where it came from (result.json / --adapt-thr)
        if EE is not None: timing.update(ee=list(EE_LOG), ee_mean_steps=round(sum(r_["mean_steps"] for r_ in EE_LOG) / max(1, len(EE_LOG)), 4), ee_eps=a.ee_eps, ee_head=a.ee_head)   # [10/19 EXP 3]
        if SDEC is not None:   # [10/17 --timed-decode] extra fields only (nothing above changes): frame_ready = [(frame, chunk, wall time)], chunk_decode = per-chunk side-stream decode, decode_s = their sum
            t1 = now(); cd = SDEC.finish(); fr = sorted(SDEC.ready); last = {}
            for f_, c_, t_ in fr: last[c_] = max(last.get(c_, 0.0), t_)
            for r_ in cd: r_["ready_t"] = last.get(r_["c"]); r_["wall_s"] = round(last[r_["c"]] - r_["push_t"], 4) if r_["c"] in last else None
            timing.update(timed_decode=1, prep_s=round(prep_s, 4), t_gen_start=t0, frame_ready=fr, frames_streamed=SDEC.nf, chunk_decode=cd, decode_s=round(sum(r_["gpu_ms"] for r_ in cd) / 1000.0, 4), finish_wait_s=round(now() - t1, 4),
                          ttff_s=round(prep_s + fr[0][2] - t0, 4) if fr else None, e2e_s=round(prep_s + (last[max(last)] - t0), 4) if last else None, peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / GiB, 3), peak_reserved_GiB=round(torch.cuda.max_memory_reserved() / GiB, 3))
            SDEC = None
        finish_clip(task, chunks, timing); SUMM["clips"][task] = {k: timing[k] for k in ("steady_chunk_s", "fps", "mean_chunk_s", "fps_mean", "n_ext", "clip_s", "peak_GiB")}; log("clip %s %.1fs steady %.3f s/chunk, mean %.3f, extended %s" % (task, timing["clip_s"], timing["steady_chunk_s"], timing["mean_chunk_s"], timing["n_ext"]))
        json.dump(SUMM, open(a.out + "/summary.json", "w"), indent=1)
elif a.role == "cloud":
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); srv.bind(("0.0.0.0", a.port)); srv.listen(1); log("listening on", a.port)
    N = prepare(TASKS[0]); warmup(N); log("warm; waiting for the device")
    conn, addr = srv.accept(); conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); log("device connected from", addr); q = queue.Queue(); threading.Thread(target=rx_thread, args=(conn, q), daemon=True).start()
    send(conn, ("READY", "")); pending = []
    while True:
        m = pending.pop(0) if pending else q.get()
        if m[0] == "EOF": log("device gone", m[1]); break
        if m[0] == "END": log("all clips done"); break
        assert m[0] == "CLIP", m[0]; task, N = m[1], m[2]; Np = prepare(task); assert Np == N, (Np, N); send(conn, ("PREPARED", task)); g = gen(a.seed, 0, 1); gcr = gen(a.seed, 0, 1); chunks = [None] * N; rec = []; nc = 0; t_clip0 = now()
        while nc < N:
            m = q.get()
            if m[0] == "EOF": raise SystemExit("device gone")
            if m[0] != "DRAFT": pending.append(m); continue
            _, c, xd, t_show, t_send = m[:5]; refine_flag = m[5] if len(m) > 5 else True; t_rx = now(); delay_until(t_send); backlog = q.qsize(); xd_raw = xd.to(dev).float(); xd = xd_raw
            if tuple(xd.shape[2:]) != tuple(Tg._chunk_shape[2:]):   # lower-resolution draft: bicubic upsample of each latent frame to the cloud's grid
                xd = torch.nn.functional.interpolate(xd.permute(1, 0, 2, 3), size=tuple(Tg._chunk_shape[2:]), mode="bicubic", align_corners=False).permute(1, 0, 2, 3).contiguous()
            eps_c = Tg.randn_like(xd, gcr); sync(); t1 = now()
            if not refine_flag:   # the device's gate accepted the draft: commit it as is (refined locally if a refiner is given), nothing to return
                x = REF.refine(up(xd_raw.half().float(), tuple(Tg._chunk_shape[2:]))[None], eps_c[None])[0].float() if REF is not None else xd; commit(x, c, "separate"); mode = "accept"; r = 0.0
            elif backlog > a.max_lag or len(USE) == 0:   # cannot keep up: take the draft as is
                x = REF.refine(up(xd_raw.half().float(), tuple(Tg._chunk_shape[2:]))[None], eps_c[None])[0].float() if REF is not None else xd; commit(x, c, "separate"); mode = "skip"; r = 0.0
            else:
                REC = [] if a.dump_pairs else None; x = denoise(c, g, start_x=xd, eps0=eps_c); commit(x, c); mode = "refine"; r = rel(x, xd)
                if a.dump_pairs: PAIRS.append(dict(c=c, xd=xd.half().cpu(), eps=REC[0].half().cpu(), x=x.half().cpu())); REC = None
            sync(); t2 = now(); chunks[c] = x; nc += 1
            if mode == "refine": send(conn, ("CORR", c, x.half().cpu(), r, t2, task))
            rec.append(dict(c=c, mode=mode, rel=round(r, 4), backlog=backlog, wait_s=round(t1 - t_rx, 4), work_s=round(t2 - t1, 4), lat_from_show_s=round(t2 - t_show, 4), net_s=round(t_rx - t_send, 4)))
        ref = [r_["work_s"] for r_ in rec if r_["mode"] == "refine"]; ref_s = sorted(ref)
        timing = dict(task=task, chunks=rec, n_refined=len(ref), n_skipped=N - len(ref), n_accepted=sum(1 for r_ in rec if r_["mode"] == "accept"), refine_s_median=(ref_s[len(ref_s) // 2] if ref_s else None), cloud_lat_from_show_median=sorted(r_["lat_from_show_s"] for r_ in rec)[N // 2], clip_s=round(now() - t_clip0, 1), peak_GiB=round(torch.cuda.max_memory_allocated() / GiB, 2))
        if a.dump_pairs: os.makedirs(a.out + "/" + task, exist_ok=True); torch.save(dict(pairs=PAIRS, task=task, seed=a.seed, nodes=NODES, use=USE, sigmas=SIG), a.out + "/" + task + "/pairs.pt"); PAIRS = []
        send(conn, ("CLIP_DONE", task))   # before the decode, so the device's late-correction wait ends with the last correction
        finish_clip(task, chunks, timing); SUMM["clips"][task] = {k: timing[k] for k in ("n_refined", "n_skipped", "refine_s_median", "cloud_lat_from_show_median", "clip_s", "peak_GiB")}; json.dump(SUMM, open(a.out + "/summary.json", "w"), indent=1)
        log("clip %s: refined %d / %d, refine %.3f s, cloud-stream latency from show %.2f s" % (task, len(ref), N, timing["refine_s_median"] or -1, timing["cloud_lat_from_show_median"]))
        send(conn, ("READY", task))   # both sides decoded: the next clip starts with an empty queue
    conn.close(); srv.close()
else:   # device
    host, port = a.connect.rsplit(":", 1); N = prepare(TASKS[0]); warmup(N)
    sock = socket.create_connection((host, int(port))); sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); log("connected to", a.connect); q = queue.Queue(); threading.Thread(target=rx_thread, args=(sock, q), daemon=True).start()
    def wait_for(kind):
        while True:
            m = q.get()
            if m[0] == "EOF": raise SystemExit("cloud gone " + m[1])
            if m[0] == kind: return
    for task in TASKS:
        wait_for("READY"); N = prepare(task); send(sock, ("CLIP", task, N)); wait_for("PREPARED")   # the cloud has prepared the clip too: no start-up backlog
        g = gen(a.seed, 0); chunks = []; rec = []; t_show = {}; corr_log = []; applied = 0; t_clip0 = now(); FEATS = []; gc = gen(a.seed, 0, 1); hw = cloud_grid(task) if REF is not None else None; CUR["task"] = task
        if ADAPT is not None: ADAPT["log"] = []
        EE_LOG.clear()
        def drain(c_now, block_s=0.0):
            """apply corrections that arrived; c_now = chunks drafted so far"""
            global applied
            t_end = now() + block_s
            while True:
                try: m = q.get(timeout=max(0.0, t_end - now()) if block_s > 0 else 0.0)
                except queue.Empty: return
                if m[0] == "EOF": raise SystemExit("cloud gone " + m[1])
                if m[0] == "CLIP_DONE": return "done"
                if m[0] != "CORR" or m[5] != task: continue
                _, c, xc, r, t_send, _task = m; t_rx = now(); delay_until(t_send); lag = now() - t_show[c]; age = c_now - 1 - c   # chunks drafted after c
                ok = r > a.thr and age < slots() and c_now < N; t1 = now()
                if ok:
                    xc = xc.to(dev).float()
                    if tuple(xc.shape[2:]) != tuple(Tg._chunk_shape[2:]): xc = torch.nn.functional.interpolate(xc.permute(1, 0, 2, 3), size=tuple(Tg._chunk_shape[2:]), mode="bicubic", align_corners=False).permute(1, 0, 2, 3).contiguous()
                    recommit(xc, c); sync(); applied += 1
                corr_log.append(dict(c=c, rel=round(r, 4), lag_s=round(lag, 3), age_chunks=age, applied=ok, recommit_s=round(now() - t1, 4) if ok else 0.0, net_s=round(t_rx - t_send, 4)))
                if block_s <= 0 and q.empty(): return
        for c in range(N):
            sync(); t1 = now()
            sc, refine_flag = None, True
            if a.draft == "model":
                fe = [] if (a.dump_feat or a.gate == "head" or ADAPT is not None) else None
                def _early(j, xs):   # B: the step-(send_step) x0 estimate goes to the cloud now; the finished draft is not sent
                    if j == a.send_step - 1: send(sock, ("DRAFT", c, xs.half().cpu(), now(), now(), True))
                x = ee_draft(c) if EE is not None else denoise(c, g, feats=fe, on_step=_early if a.send_step else None, adapt=ADAPT is not None); commit(x, c)
                if a.dump_feat: FEATS.append(torch.stack(fe))
                if a.gate != "none": sc, refine_flag = gate_score(task, c, x, fe, chunks[-1] if chunks else None)
            elif a.draft == "prev": x = chunks[-1] if chunks else Tg.chunk_noise(c, g).to(dev).float()
            else: x = Tg.chunk_noise(c, g).to(dev).float()
            shown = refine_local(x, c, hw, gc) if REF is not None else x
            sync(); t2 = now(); t_show[c] = t2; chunks.append(shown)
            if not a.send_step: send(sock, ("DRAFT", c, x.half().cpu(), t2, now(), refine_flag))
            t3 = now(); drain(c + 1); sync(); t4 = now()
            rec.append(dict(c=c, draft_s=round(t2 - t1, 4), send_s=round(t3 - t2, 4), corr_s=round(t4 - t3, 4), total_s=round(t4 - t1, 4), gate=sc, refine=refine_flag, ext=(ADAPT["log"][-1]["ext"] if ADAPT is not None and a.draft == "model" else None)))
        # late corrections (no chunk left to benefit; logged for the lag statistics) and the cloud's end-of-clip marker
        t_e = now() + a.wait_s
        while now() < t_e:
            if drain(N, block_s=max(0.01, t_e - now())) == "done": break
        tot = sorted(r_["total_s"] for r_ in rec[3:]); dr = sorted(r_["draft_s"] for r_ in rec[3:]); lags = [c_["lag_s"] for c_ in corr_log]
        timing = dict(task=task, chunks=rec, corrections=corr_log, n_corr_received=len(corr_log), n_applied=applied, n_sent=sum(1 for r_ in rec if r_["refine"]), draft_s_median=dr[len(dr) // 2], total_s_median=tot[len(tot) // 2], fps_shown=round(12.0 / tot[len(tot) // 2], 2),
                      corr_lag_median=(sorted(lags)[len(lags) // 2] if lags else None), clip_s=round(now() - t_clip0, 1), peak_GiB=round(torch.cuda.max_memory_allocated() / GiB, 2))
        if ADAPT is not None: timing.update(adapt=list(ADAPT["log"]), adapt_thr=ADAPT["thr"], adapt_thr_source=ADAPT["thr_source"])   # [10/19 audit] the device stream now keeps the step-1 head scores (sc) too, not only "ext" per chunk, + the effective threshold and its source
        if EE is not None: timing.update(ee=list(EE_LOG), ee_mean_steps=round(sum(r_["mean_steps"] for r_ in EE_LOG) / max(1, len(EE_LOG)), 4), ee_eps=a.ee_eps, ee_head=a.ee_head)   # [10/19 EXP 3]
        if a.dump_feat: os.makedirs(a.out + "/" + task, exist_ok=True); torch.save(dict(h=torch.stack(FEATS), task=task, seed=a.seed, nodes=NODES, use=USE, area=a.draft_area or a.area), a.out + "/" + task + "/feat.pt")
        finish_clip(task, chunks, timing); SUMM["clips"][task] = {k: timing[k] for k in ("n_corr_received", "n_applied", "draft_s_median", "total_s_median", "fps_shown", "corr_lag_median", "clip_s", "peak_GiB")}; json.dump(SUMM, open(a.out + "/summary.json", "w"), indent=1)
        log("clip %s: draft %.3f s, total %.3f s/chunk = %.1f fps shown, corrections %d received / %d applied, lag %.2f s" % (task, timing["draft_s_median"], timing["total_s_median"], timing["fps_shown"], len(corr_log), applied, timing["corr_lag_median"] or -1))
    send(sock, ("END",)); sock.close()
log("DONE", a.out)
