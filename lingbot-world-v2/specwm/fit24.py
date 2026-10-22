"""Does the 14B fit on a device GPU? Memory footprint and speed of the 14B alone (no draft model, no cloud) with the optimized stack,
optionally with INT4 weights and an FP8 KV window, under an allocator cap that emulates a 24 / 32 GB device.
Same generation loop as res_split_infer --fused --partial-commit 1 --no-device (the quality / realtime modes), one GPU.
Writes <out>/fit.json: weight bytes by component, KV bytes, resident after prepare, peak during generation, peak during per-frame streamed
VAE decode, steady s/chunk, fps, and the stage at which an OOM happened (if any). Optional lat.pt / spec.mp4 for a quality check.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.fit24 --task easy_lake --area 270336 --wq fp8 --kv bf16 --cap 24 --out DIR"""
import argparse, os, sys, json, time, gc, torch, torch.nn as nn
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--task", default="easy_lake", help="task measured; with --tasks the remaining tasks are generated afterwards (videos only, for a quality check)"); ap.add_argument("--tasks", default=""); ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--frames", type=int, default=161)
ap.add_argument("--area", type=int, default=270336, help="max_area: 832x480=399360, 704x384=270336, 576x320=184320"); ap.add_argument("--nodes", default="0,500", help="denoising nodes (0,500 = 2-step; 0,250,500,750 = teacher 4-step)")
ap.add_argument("--wq", default="fp8", choices=["bf16", "fp8", "int4", "awq", "gptq4"], help="weights of the 14B blocks (int4 = round-to-nearest with scale search, awq = activation-aware INT4, gptq4 = GPTQ 4-bit group 128 with Hessians from 3 calibration chunks; all packed, dequantized on the fly)"); ap.add_argument("--int4-kernel", default="torch", choices=["torch", "triton"], help="dequant of int4 / gptq4 weights: the stack/reshape chain or the fused Triton kernel (int4_kernels.py)"); ap.add_argument("--kv", default="bf16", choices=["bf16", "fp8"], help="dtype of the self-attention KV window")
ap.add_argument("--model", default="14b", choices=["14b", "1p3b"], help="1p3b = the LingBot 1.3B causal-fast draft (same stack; assets from the 14B dir)"); ap.add_argument("--window", type=int, default=18, help="local_attn (latent frames in the KV window, incl. the 6 sink frames)")
ap.add_argument("--commit", default="pc1", choices=["pc1", "fused", "separate"], help="pc1 = fused + partial commit 1 (ours); separate = clean commit pass (teacher)")
ap.add_argument("--cap", type=float, default=0, help="GB; cap the CUDA caching allocator (device emulation, OOM = does not fit). 0 = no cap")
ap.add_argument("--decode", default="stream", choices=["stream", "tiled"], help="device decode measured: per-latent-frame streamed causal decode, or spatial 2x2 tiled chunk decode (vae_tiled, approximate)"); ap.add_argument("--compile", type=int, default=1); ap.add_argument("--video", type=int, default=0, help="1: full decode -> spec.mp4 after the measurement (cap lifted)"); ap.add_argument("--out", required=True)
a = ap.parse_args(); a.out = os.path.abspath(a.out)   # resolved before the chdir below (envrun.sh already cd's into lingbot-world-v2: pass absolute paths)
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
GiB = 2 ** 30; dev = "cuda:0"; sync = torch.cuda.synchronize; os.makedirs(a.out, exist_ok=True)
R = dict(vars(a)); R["gpu"] = torch.cuda.get_device_name(0); R["gpu_total_GiB"] = round(torch.cuda.get_device_properties(0).total_memory / GiB, 1); R["oom_stage"] = None
def save():
    json.dump(R, open(a.out + "/fit.json", "w"), indent=1)
def mem(tag):
    sync(); R["alloc_" + tag + "_GiB"] = round(torch.cuda.memory_allocated() / GiB, 2); R["reserved_" + tag + "_GiB"] = round(torch.cuda.memory_reserved() / GiB, 2)
    print("[mem] %-16s alloc %.2f  reserved %.2f GiB" % (tag, R["alloc_" + tag + "_GiB"], R["reserved_" + tag + "_GiB"]), flush=True)
def nbytes(m): return sum(p.numel() * p.element_size() for p in m.parameters()) + sum(b.numel() * b.element_size() for b in m.buffers())
def apply_cap():   # device emulation: the allocator may not grow past the cap (the CUDA context and cuBLAS / compile workspaces live outside it).
    # Applied AFTER the weight swap: the bf16 checkpoint is loaded to the GPU first (34.5 GiB); a device would load the quantized weights directly.
    if a.cap > 0:
        gc.collect(); torch.cuda.empty_cache(); tot = torch.cuda.get_device_properties(0).total_memory / GiB; torch.cuda.set_per_process_memory_fraction(min(1.0, a.cap / tot)); print("[cap] %.0f GB of %.0f GB" % (a.cap, tot), flush=True)
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
stage = "load"
try:
    if a.model == "1p3b": Tg = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=a.window, swap_models=False, build_draft=False, timesteps_index=tuple(int(v) for v in a.nodes.split(",")), cfg_name="i2v-1.3B", assets_dir=FAST)
    else: Tg = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=a.window, swap_models=False, build_draft=False, timesteps_index=tuple(int(v) for v in a.nodes.split(",")))
    tm = Tg.target_model; R["dit_bf16_GiB"] = round(nbytes(tm) / GiB, 2)
    FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True)
    stage = "quantize"
    if a.wq == "fp8": FP.fp8_swap(tm, True, keep=False)
    elif a.wq == "int4":
        from specwm.quantize import quantize_packed_int4
        t_ = time.time(); quantize_packed_int4(tm, group=128, search=True, verbose=True); R["quant_s"] = round(time.time() - t_, 1)
        if a.int4_kernel == "triton":
            from specwm.quantize import PackedInt4Linear
            from specwm.int4_kernels import PackedInt4LinearT
            for n_, m_ in list(tm.named_modules()):
                for cn, ch in list(m_.named_children()):
                    if isinstance(ch, PackedInt4Linear): setattr(m_, cn, PackedInt4LinearT.from_packed(ch))
    gc.collect(); torch.cuda.empty_cache()   # awq is calibrated after prepare (needs forwards), see below
    if a.wq not in ("awq", "gptq4"): apply_cap()
    R["dit_GiB"] = round(nbytes(tm) / GiB, 2); R["vae_GiB"] = round(nbytes(Tg.pipe.vae.model) / GiB, 2)
    mem("after_load"); R["other_resident_GiB"] = round(R["alloc_after_load_GiB"] - R["dit_GiB"] - R["vae_GiB"], 2)   # CLIP image encoder etc.
    if a.kv == "fp8":   # the ring buffer is written with an implicit cast and read back through _sdpa's half(), so only the allocation changes
        _init = Tg.pipe._initialize_self_kv_cache
        Tg.pipe._initialize_self_kv_cache = lambda num_layers, shape, dtype, device: _init(num_layers=num_layers, shape=shape, dtype=torch.float8_e4m3fn, device=device)
    if a.compile:
        import torch._dynamo as _dyn; _dyn.config.recompile_limit = 512; _dyn.config.accumulated_recompile_limit = 8192
        if a.wq not in ("awq", "gptq4"):
            for blk in tm.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
    enc = Tg.pipe.text_encoder; _cache = {}
    class _C:
        def __init__(s): s.model = enc.model
        def __call__(s, texts, device):
            k = (tuple(texts), str(device))
            if k not in _cache: _cache[k] = enc(texts, device)
            return _cache[k]
    Tg.pipe.text_encoder = _C()
    SIG = [float(s) for s in Tg.node_sigmas()]; T = ALL[a.task]
    def gen(seed, c): return torch.Generator().manual_seed(seed * 1009 + c * 7)
    stage = "prepare"
    t0 = time.time(); cond = Tg.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=a.frames, max_area=a.area); Tg.reset_caches(); N = cond["num_chunks"]; sync(); R["prepare_s"] = round(time.time() - t0, 2)
    kvb = sum(c["k"].numel() * c["k"].element_size() + c["v"].numel() * c["v"].element_size() for c in Tg._caches["target"]["kv"]); crb = sum(c["k"].numel() * c["k"].element_size() * 2 for c in Tg._caches["target"]["cross"])
    R.update(chunks=N, lat_shape=list(Tg._chunk_shape), frame_seqlen=Tg._frame_seqlen, kv_tokens=int(Tg._caches["target"]["kv"][0]["k"].shape[1]), kv_dtype=str(Tg._caches["target"]["kv"][0]["k"].dtype), kv_GiB=round(kvb / GiB, 2), cross_GiB=round(crb / GiB, 2))
    print("[kv] %d tokens x %d layers, %s: %.2f GiB (+ cross %.2f)" % (R["kv_tokens"], len(Tg._caches["target"]["kv"]), R["kv_dtype"], R["kv_GiB"], R["cross_GiB"]), flush=True)
    mem("after_prepare")
    @torch.no_grad()
    def one_chunk(c, g, commit=None):
        z = Tg.chunk_noise(c, g); x = None
        for i in range(len(SIG)):
            if i > 0: s_ = SIG[i]; z = (1 - s_) * x + s_ * Tg.randn_like(x, g)
            x = Tg.target_x0(z, i, c).float()
        if (commit or a.commit) == "pc1":
            fs = Tg._frame_seqlen; L = Tg._chunk_size * fs; FP.ACTIVE["a"] = dict(idx=torch.arange(L - fs, L, device=dev), n=fs, L=L)
            try: Tg.commit(x, c)
            finally: FP.ACTIVE["a"] = None
        elif a.commit == "fused": Tg._committed_frames += Tg._chunk_size
        else: Tg.commit(x, c)
        return x
    if a.wq == "gptq4":   # GPTQ 4-bit: Hessians X^T X per linear from 3 chunks of the bf16 model (uncompiled), then per-layer GPTQ, packed storage
        from specwm.gptq import HessianCollector, GPTQLinear
        from specwm.int4_kernels import PackedInt4LinearT
        stage = "gptq_calibrate"; t_ = time.time(); hc = HessianCollector(tm, min_features=1024); g0 = gen(a.seed + 2, 0)
        for c in range(3): one_chunk(c, g0, commit="separate")
        Hs = hc.finish(); Tg.reset_caches(); nq = 0
        for n_ in [n for n, m in tm.named_modules() if isinstance(m, nn.Linear) and m.in_features >= 1024]:
            par = tm.get_submodule(n_.rsplit(".", 1)[0]) if "." in n_ else tm; leaf = n_.rsplit(".", 1)[-1]; lin = getattr(par, leaf)
            H_ = Hs.pop(n_); gl = GPTQLinear(lin, H_, bits=4, group=128)
            if not torch.isfinite(gl.scale.float()).all():   # Cholesky failed for this layer at the default damping (seen at 832: blocks.1.self_attn.o) -> re-run with 10x damping
                import specwm.gptq as _gq; _orig = _gq.gptq_quantize_weight; _gq.gptq_quantize_weight = lambda W, H, bits=4, group=128, blocksize=128, percdamp=0.01: _orig(W, H, bits=bits, group=group, blocksize=blocksize, percdamp=0.1)
                try: gl = GPTQLinear(lin, H_, bits=4, group=128)
                finally: _gq.gptq_quantize_weight = _orig
                print("[gptq4] %s: non-finite at damp 0.01, redone at 0.1 -> finite %s" % (n_, bool(torch.isfinite(gl.scale.float()).all())), flush=True); R.setdefault("gptq_redamped", []).append(n_)
            setattr(par, leaf, PackedInt4LinearT(gl.q, gl.scale, gl.bias, gl.O, gl.I, gl.g)); del lin, gl, H_; nq += 1
        del Hs; gc.collect(); torch.cuda.empty_cache(); R["quant_s"] = round(time.time() - t_, 1); print("[gptq4] %d linears, %.0f s" % (nq, R["quant_s"]), flush=True)
        bad = [n for n, m in tm.named_modules() if isinstance(m, PackedInt4LinearT) and not torch.isfinite(m.scale.float()).all()]; R["gptq_nonfinite_layers"] = bad
        if bad: print("[gptq4] NON-FINITE scales in %d layers: %s" % (len(bad), bad[:6]), flush=True)
        R["dit_GiB"] = round(nbytes(tm) / GiB, 2); mem("after_gptq"); R["alloc_after_load_GiB"] = round(R["alloc_after_gptq_GiB"] - R["kv_GiB"] - R["cross_GiB"], 2)
        apply_cap()
        if a.compile:
            for blk in tm.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
    if a.wq == "awq":   # activation-aware INT4: mean |x| per input channel from 3 chunks of the bf16 model (uncompiled), then quantize
        from specwm.awq import ActStats, quantize_awq
        stage = "awq_calibrate"; t_ = time.time(); st = ActStats(tm, min_features=1024); g0 = gen(a.seed + 2, 0)
        for c in range(3): one_chunk(c, g0, commit="separate")
        means = st.finish(); Tg.reset_caches(); quantize_awq(tm, means, alpha=0.5, group=128, verbose=True); del means; gc.collect(); torch.cuda.empty_cache(); R["quant_s"] = round(time.time() - t_, 1)
        R["dit_GiB"] = round(nbytes(tm) / GiB, 2); mem("after_awq"); R["alloc_after_load_GiB"] = round(R["alloc_after_awq_GiB"] - R["kv_GiB"] - R["cross_GiB"], 2)
        apply_cap()
        if a.compile:
            for blk in tm.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
    stage = "warmup"   # compile every shape (KV fill 1..6 chunks x step / partial-commit) on a throw-away clip, as res_split_infer --warm 1
    g = gen(a.seed + 1, 0)
    for c in range(N): one_chunk(c, g)
    Tg.reset_caches(); gc.collect(); torch.cuda.empty_cache(); sync(); torch.cuda.reset_peak_memory_stats()
    stage = "generate"
    chunks = []; ct = []; g = gen(a.seed, 0)
    for c in range(N):
        sync(); t1 = time.time(); x = one_chunk(c, g); sync(); ct.append(round(time.time() - t1, 4)); chunks.append(x)
        if c == 0: R["chunk0_finite"] = bool(torch.isfinite(x).all()); R["chunk0_std"] = round(float(x.float().std()), 4); print("[gen] chunk 0 finite %s std %.3f" % (R["chunk0_finite"], R["chunk0_std"]), flush=True)
    R["chunk_s"] = ct; st = sorted(ct[5:]); R["steady_chunk_s"] = round(st[len(st) // 2], 3); R["fps"] = round(12.0 / R["steady_chunk_s"], 2)
    R["peak_gen_GiB"] = round(torch.cuda.max_memory_allocated() / GiB, 2); R["peak_gen_reserved_GiB"] = round(torch.cuda.max_memory_reserved() / GiB, 2)
    print("[gen] steady %.3f s/chunk = %.2f fps | peak alloc %.2f reserved %.2f GiB" % (R["steady_chunk_s"], R["fps"], R["peak_gen_GiB"], R["peak_gen_reserved_GiB"]), flush=True)
    mem("after_gen")
    stage = "decode"   # device-style streamed decode: one latent frame at a time through the causal decoder (as StreamDecoder in res_split_infer)
    v = Tg.pipe.vae; m = v.model; m.clear_cache(); torch.cuda.reset_peak_memory_stats(); dt = []
    if a.decode == "tiled":
        from specwm.vae_tiled import tiled_decode
        for c in range(min(4, N)):
            sync(); t1 = time.time()
            with torch.no_grad(): o = tiled_decode(v, chunks[c], tiles=(2, 2))
            sync(); dt.append(round(time.time() - t1, 4)); del o
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=v.dtype):
        for c in range(min(4, N) if a.decode == "stream" else 0):
            sync(); t1 = time.time(); z = chunks[c]; sc = v.scale; zz = z.unsqueeze(0) / sc[1].view(1, -1, 1, 1, 1) + sc[0].view(1, -1, 1, 1, 1); xx = m.conv2(zz)
            for i in range(xx.shape[2]):
                m._conv_idx = [0]; o = m.decoder(xx[:, :, i:i + 1], feat_cache=m._feat_map, feat_idx=m._conv_idx).float().clamp_(-1, 1); del o
            sync(); dt.append(round(time.time() - t1, 4))
    m.clear_cache(); R["decode_chunk_s"] = dt; R["peak_decode_GiB"] = round(torch.cuda.max_memory_allocated() / GiB, 2); R["peak_decode_reserved_GiB"] = round(torch.cuda.max_memory_reserved() / GiB, 2)
    print("[decode] %.3f s/chunk | peak alloc %.2f reserved %.2f GiB" % (dt[-1], R["peak_decode_GiB"], R["peak_decode_reserved_GiB"]), flush=True)
    R["peak_GiB"] = max(R["peak_gen_GiB"], R["peak_decode_GiB"]); R["peak_reserved_GiB"] = max(R["peak_gen_reserved_GiB"], R["peak_decode_reserved_GiB"]); R["fits"] = True
    save()
    if a.video:
        if a.cap > 0: torch.cuda.set_per_process_memory_fraction(1.0)
        lat = torch.cat(chunks, 1); torch.save(lat.half().cpu(), a.out + "/lat.pt"); Tg.decode_and_save([lat.to(dev)], a.out)
        for task in [t for t in a.tasks.split(",") if t and t != a.task]:   # more clips for the quality check (same seed rule as res_split_infer)
            od = a.out + "/" + task
            if os.path.exists(od + "/spec.mp4"): continue
            os.makedirs(od, exist_ok=True); T = ALL[task]; Tg.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=a.frames, max_area=a.area); Tg.reset_caches(); N = cond["num_chunks"]
            g = gen(a.seed, 0); cs = []; t1 = time.time()
            for c in range(N): cs.append(one_chunk(c, g))
            sync(); lat = torch.cat(cs, 1); Tg.decode_and_save([lat.to(dev)], od); json.dump(dict(task=task, seed=a.seed, gen_s=round(time.time() - t1, 1)), open(od + "/loop.json", "w")); print("[clip] %s %.1fs" % (task, time.time() - t1), flush=True)
except torch.OutOfMemoryError as e:
    R["oom_stage"] = stage; R["oom_msg"] = str(e).split("\n")[0][:300]; R["fits"] = False; print("[OOM at %s] %s" % (stage, R["oom_msg"]), flush=True); save()
print("[FIT_DONE]", json.dumps({k: R[k] for k in ("wq", "kv", "area", "cap", "fits", "oom_stage") if k in R}), flush=True)
