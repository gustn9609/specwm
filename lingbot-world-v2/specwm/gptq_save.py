"""[10/19 thesis] OFFLINE GPTQ-4 checkpoint of the 14B causal-fast DiT blocks (what fit24 --wq gptq4 computes in-process and throws away).
Load = fit24's: the bf16 14B on the GPU (34.5 GiB), fast_patch flags, NO compile, NO FP8 swap (Hessians come from the bf16 model). Calibration = fit24's
loop (one_chunk with commit="separate") on --chunks chunks of EACH of --tasks (default 2 StEvo scenes of specwm_out/refiner/train_tasks.txt that are
NOT frozen benchmark scenes), at --area (832x480 = 399361, the baselines' +1 convention), on the --nodes schedule (0,500 = rt2: the Hessians see the
sigma 1.0 and 0.833 inputs + the clean t=0 commit pass), KV window --window. HessianCollector accumulates H = X^T X per block linear on the host.
Then GPTQ 4-bit, group 128, per linear (specwm.gptq.GPTQLinear; a non-finite result at damp 0.01 is redone at 0.1 = fit24's fallback), packed to the
int4_kernels storage (specwm.int4_packed.pack_codes) and saved as ONE file:
  {"meta": {...calibration, resolution, seed, damp, redamped / non-finite layers, peaks...}, "layers": {"blocks.<i>.<name>": {packed uint8 [O, I/2],
   scale bf16 [O, I/G], bias bf16 | None, O, I, g}}}        (torch.load(weights_only=True) works: tensors + primitives only)
Scope = every nn.Linear under tm.blocks with in_features >= 1024 (568 / 14B: attention q,k,v,o, cross q,k,v,o, ffn) = the layers fast_patch.fp8_swap
turns into FP8Linear minus the small ones; the non-block linears (text / time embedding, head) stay bf16 as in the FP8 arm. NOT fit24's exact scope
(fit24 also quantized the non-block linears with in_features >= 1024).
GPU budget: one PRO 6000 (bf16 DiT 34.5 GiB + bf16 KV window 18 at 832x480 ~20 GiB + activations + one X^T X temporary <= 0.75 GiB; measured by
fit24's calibration at 832: < 70 GiB). The Hessians live in host RAM (~45 GiB for the block linears). Each quantized layer's bf16 weight is freed right
after its GPTQ (the GPU gets emptier as it goes). Host RAM for the packed result ~7.5 GB. Idempotent: an existing --out that loads with 0 non-finite
layers is kept (exit 0) unless --force 1. Refuses to start when /home has < --min-free-gb free (the user approves deletions).
usage: CUDA_VISIBLE_DEVICES=g /home/hk28965/specwm/scripts/envrun.sh python -m specwm.gptq_save --out /home/hk28965/specwm/specwm_out/gptq4/14b_g128_832.pt
prints [GPTQ_SAVE_DONE] {...} on success (exit 0); exit 3 = disk, 4 = non-finite layers remain, 5 = reload check failed."""
import argparse, os, sys, json, time, gc, shutil, torch, torch.nn as nn
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=S + "/specwm_out/gptq4/14b_g128_832.pt"); ap.add_argument("--tasks", default="ball_roll_down,bouncing_ball_rebound", help="calibration task ids (data/stevo/tasks_camera.json, + --tasks-json); NOT evaluation scenes")
ap.add_argument("--tasks-json", default="", help="extra task list (SPECWM_TASKS schema) merged over tasks_camera.json"); ap.add_argument("--area", type=int, default=399361, help="832x480 with the baselines' +1 convention")
ap.add_argument("--nodes", default="0,500", help="denoising schedule during calibration (0,500 = rt2)"); ap.add_argument("--window", type=int, default=18); ap.add_argument("--chunks", type=int, default=3, help="calibration chunks per task")
ap.add_argument("--frames", type=int, default=161); ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--group", type=int, default=128); ap.add_argument("--bits", type=int, default=4)
ap.add_argument("--damp", type=float, default=0.01, help="GPTQ percdamp; a non-finite layer is redone at 10x (fit24)"); ap.add_argument("--min-free-gb", type=float, default=20.0); ap.add_argument("--force", type=int, default=0)
a = ap.parse_args(); a.out = os.path.abspath(a.out)
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.int4_packed import pack_codes, check_file
GiB = 2 ** 30; dev = "cuda:0"; sync = torch.cuda.synchronize; os.makedirs(os.path.dirname(a.out), exist_ok=True); now = time.time
def log(*m): print("[gptq_save %s]" % time.strftime("%H:%M:%S"), *m, flush=True)
free_gb = shutil.disk_usage(os.path.dirname(a.out)).free / 1e9
if free_gb < a.min_free_gb: log("only %.1f GB free under %s (< %.0f GB): refusing to write ~7.5 GB; free space first (the user approves deletions)" % (free_gb, os.path.dirname(a.out), a.min_free_gb)); sys.exit(3)
if os.path.exists(a.out) and not a.force:
    try:
        meta, nl, bad = check_file(a.out)
        if not bad: log("exists and loads: %d layers, 0 non-finite -> nothing to do (--force 1 to redo)" % nl); print("[GPTQ_SAVE_DONE]", json.dumps(dict(out=a.out, n_layers=nl, nonfinite=0, reused=True)), flush=True); sys.exit(0)
        log("exists but %d non-finite layers (%s) -> redoing" % (len(bad), bad[:4]))
    except Exception as e: log("exists but does not load (%s) -> redoing" % e)
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
from specwm.gptq import HessianCollector, GPTQLinear
import specwm.gptq as _gq
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
if a.tasks_json: ALL.update({t["id"]: t for t in json.load(open(a.tasks_json))})
TASKS = [t for t in a.tasks.split(",") if t]; assert all(t in ALL for t in TASKS), [t for t in TASKS if t not in ALL]
FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"; NODES = tuple(int(v) for v in a.nodes.split(","))
META = dict(model="lingbot-world-v2-14b-causal-fast", ckpt_dir=FAST, bits=a.bits, group=a.group, damp=a.damp, calib_tasks=TASKS, calib_tasks_source="specwm_out/refiner/train_tasks.txt (StEvo scenes used for the 10/16 refiner pairs; none is a frozen benchmark scene)",
            calib_chunks_per_task=a.chunks, nodes=list(NODES), use=list(range(len(NODES))), commit="separate (clean t=0 commit pass included in the Hessians)", window=a.window, area=a.area, frames=a.frames, seed=a.seed,
            seed_rule="gen(seed + 2, 0) = fit24's calibration generator", scope="tm.blocks nn.Linear with in_features >= 1024 (non-block linears stay bf16)", hessian="X^T X / n_tokens per linear, accumulated on the host over both tasks (specwm.gptq.HessianCollector)",
            storage="int4_kernels.PackedInt4LinearT: packed uint8 [O, I/2] (even col low nibble, codes + 8), scale bf16 [O, I/g]; bias bf16", gpu=torch.cuda.get_device_name(0), gpu_total_GiB=round(torch.cuda.get_device_properties(0).total_memory / GiB, 1),
            torch=str(torch.__version__), made=time.strftime("%F %T"), script="specwm/gptq_save.py [10/19 thesis]")
t_all = now()
Tg = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=a.window, swap_models=False, build_draft=False, timesteps_index=NODES); tm = Tg.target_model
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True)
enc = Tg.pipe.text_encoder; _cache = {}
class _C:
    def __init__(s): s.model = enc.model
    def __call__(s, texts, device):
        k = (tuple(texts), str(device))
        if k not in _cache: _cache[k] = enc(texts, device)
        return _cache[k]
Tg.pipe.text_encoder = _C(); SIG = [float(s) for s in Tg.node_sigmas()]
def gen(seed, c): return torch.Generator().manual_seed(seed * 1009 + c * 7)
sync(); META["load_s"] = round(now() - t_all, 1); META["peak_load_GiB"] = round(torch.cuda.max_memory_allocated() / GiB, 2); log("loaded bf16 14B in %.0f s, alloc %.1f GiB; nodes %s sigmas %s" % (META["load_s"], torch.cuda.memory_allocated() / GiB, NODES, [round(s, 3) for s in SIG]))
@torch.no_grad()
def one_chunk(c, g):   # = fit24.one_chunk(commit="separate"): every node of the schedule from noise, then the clean commit pass
    z = Tg.chunk_noise(c, g); x = None
    for i in range(len(SIG)):
        if i > 0: s_ = SIG[i]; z = (1 - s_) * x + s_ * Tg.randn_like(x, g)
        x = Tg.target_x0(z, i, c).float()
    Tg.commit(x, c); return x
# ---- calibration: Hessians of the block linears only (names relative to tm.blocks -> "blocks.<i>...") ----
hc = HessianCollector(tm.blocks, min_features=1024); t_cal = now(); META["calib"] = []
for ti, task in enumerate(TASKS):
    T = ALL[task]; t0 = now(); cond = Tg.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=a.frames, max_area=a.area); Tg.reset_caches(); N = cond["num_chunks"]
    if ti == 0: META["lat_shape"] = list(Tg._chunk_shape); META["resolution"] = "%dx%d" % (Tg._chunk_shape[3] * 8, Tg._chunk_shape[2] * 8); META["frame_seqlen"] = Tg._frame_seqlen
    g0 = gen(a.seed + 2, 0); xs = []
    for c in range(min(a.chunks, N)): xs.append(one_chunk(c, g0))
    sync(); fin = all(bool(torch.isfinite(x).all()) for x in xs); META["calib"].append(dict(task=task, chunks=len(xs), s=round(now() - t0, 1), finite=fin, x0_std=round(float(xs[-1].float().std()), 4)))
    log("calibration task %s: %d chunks in %.0f s (finite %s), peak alloc %.1f GiB" % (task, len(xs), now() - t0, fin, torch.cuda.max_memory_allocated() / GiB)); del xs; Tg.reset_caches()
Hs = hc.finish(); META["calib_s"] = round(now() - t_cal, 1); META["hessian_tokens"] = {k: int(v) for k, v in list(hc.n.items())[:3]}; META["n_hessians"] = len(Hs)
META["peak_calib_GiB"] = round(torch.cuda.max_memory_allocated() / GiB, 2); log("%d Hessians in %.0f s (tokens per layer e.g. %s), host RSS ~%.0f GiB of Hessians" % (len(Hs), META["calib_s"], META["hessian_tokens"], sum(h.numel() * 4 for h in Hs.values()) / GiB))
Tg._caches = {}; gc.collect(); torch.cuda.empty_cache()
# ---- GPTQ per block linear, packed, freed on the GPU as we go ----
LAYERS = {}; redamped = []; nonfinite = []; missing = []; t_q = now(); nq = 0
names = ["blocks." + n for n, m in tm.blocks.named_modules() if isinstance(m, nn.Linear) and m.in_features >= 1024]
_orig = _gq.gptq_quantize_weight
for n_ in names:
    par = tm.get_submodule(n_.rsplit(".", 1)[0]); leaf = n_.rsplit(".", 1)[-1]; lin = getattr(par, leaf); H_ = Hs.pop(n_[len("blocks."):], None)
    if H_ is None: missing.append(n_); H_ = torch.eye(lin.in_features)
    assert lin.in_features % a.group == 0, (n_, lin.in_features)
    gl = GPTQLinear(lin, H_, bits=a.bits, group=a.group) if a.damp == 0.01 else None
    if gl is None:
        _gq.gptq_quantize_weight = lambda W, H, bits=4, group=128, blocksize=128, percdamp=0.01: _orig(W, H, bits=bits, group=group, blocksize=blocksize, percdamp=a.damp)
        try: gl = GPTQLinear(lin, H_, bits=a.bits, group=a.group)
        finally: _gq.gptq_quantize_weight = _orig
    if not torch.isfinite(gl.scale.float()).all():   # fit24's fallback: 10x damping
        _gq.gptq_quantize_weight = lambda W, H, bits=4, group=128, blocksize=128, percdamp=0.01: _orig(W, H, bits=bits, group=group, blocksize=blocksize, percdamp=a.damp * 10)
        try: gl = GPTQLinear(lin, H_, bits=a.bits, group=a.group)
        finally: _gq.gptq_quantize_weight = _orig
        ok = bool(torch.isfinite(gl.scale.float()).all()); redamped.append(n_); log("%s: non-finite at damp %g, redone at %g -> finite %s" % (n_, a.damp, a.damp * 10, ok))
        if not ok: nonfinite.append(n_)
    packed, sc = pack_codes(gl.q, gl.scale, gl.O, gl.I, gl.g)
    LAYERS[n_] = dict(packed=packed.cpu(), scale=sc.cpu(), bias=(gl.bias.detach().to(torch.bfloat16).cpu() if gl.bias is not None else None), O=int(gl.O), I=int(gl.I), g=int(gl.g))
    setattr(par, leaf, nn.Identity()); del lin, gl, H_, packed, sc; nq += 1
    if nq % 40 == 0: gc.collect(); torch.cuda.empty_cache(); log("%d / %d layers, %.0f s, alloc %.1f GiB" % (nq, len(names), now() - t_q, torch.cuda.memory_allocated() / GiB))
META.update(quant_s=round(now() - t_q, 1), n_layers=nq, redamped=redamped, nonfinite=nonfinite, no_hessian=missing, damp_bumps=getattr(_orig, "damp_bumps", 0), peak_GiB=round(torch.cuda.max_memory_allocated() / GiB, 2),
            bytes_packed=int(sum(L["packed"].numel() + L["scale"].numel() * 2 + (L["bias"].numel() * 2 if L["bias"] is not None else 0) for L in LAYERS.values())), total_s=round(now() - t_all, 1))
log("GPTQ %d layers in %.0f s; redamped %d %s; non-finite %d; %.2f GB packed" % (nq, META["quant_s"], len(redamped), redamped[:4], len(nonfinite), META["bytes_packed"] / 1e9))
tmp = a.out + ".tmp"; torch.save(dict(meta=META, layers=LAYERS), tmp); os.replace(tmp, a.out); json.dump(META, open(a.out[:-3] + ".meta.json", "w"), indent=1)
del LAYERS; gc.collect()
try: meta2, nl, bad = check_file(a.out)
except Exception as e: log("reload check FAILED: %s" % e); sys.exit(5)
log("saved %s (%.2f GB on disk), reload: %d layers, %d non-finite" % (a.out, os.path.getsize(a.out) / 1e9, nl, len(bad)))
print("[GPTQ_SAVE_DONE]", json.dumps(dict(out=a.out, n_layers=nl, nonfinite=len(bad), redamped=len(redamped), quant_s=META["quant_s"], total_s=META["total_s"], peak_GiB=META["peak_GiB"], resolution=META.get("resolution"), tasks=TASKS)), flush=True)
sys.exit(4 if bad else 0)
