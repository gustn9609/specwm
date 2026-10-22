"""Kernel breakdown of one 14B forward with and without Ulysses SP (FP8, compile, SageAttention, ring KV), after K committed chunks.
Run under torchrun with 1 or 2 processes; rank 0 prints wall ms and the CUDA time per category (gemm / attention / all-to-all & gather / other).
usage: torchrun --nproc_per_node P -m specwm.bench_sp [--sp]"""
import os, sys, json, time, collections, argparse, numpy as np, torch, torch.distributed as dist
S = "/home/hk28965/specwm"; ap = argparse.ArgumentParser(); ap.add_argument("--sp", action="store_true"); ap.add_argument("--K", type=int, default=7); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
LR = int(os.environ.get("LOCAL_RANK", 0)); torch.cuda.set_device(LR); dist.init_process_group("nccl", device_id=torch.device("cuda", LR)); R0 = dist.get_rank() == 0; DEV = "cuda:%d" % LR
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
T = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}["easy_urban_rain"]; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 500), device=DEV)
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True)
if a.sp: from specwm import fast_sp as FSP; FSP.apply_sp()
FP.fp8_swap(B.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 64
for blk in B.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
B.prepare(T["image"], "", T["action"], frame_num=161, max_area=480 * 832); B.reset_caches()
if a.sp: FSP.local_head_caches(B)
g = torch.Generator().manual_seed(0)
with torch.no_grad():
    for ci in range(a.K): x = B.chunk_noise(ci, g); B.commit(B.target_x0(x, 0, ci), ci)
xin = B.chunk_noise(a.K, g)
def fwd(): return B.target_x0(xin, 0, a.K)
with torch.no_grad():
    for _ in range(4): fwd()
    torch.cuda.synchronize(); dist.barrier(); ts = []
    for _ in range(6): t = time.time(); fwd(); torch.cuda.synchronize(); ts.append(time.time() - t)
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as P: fwd(); torch.cuda.synchronize()
def cat(n):
    l = n.lower()
    if "nccl" in l or "alltoall" in l or "all_to_all" in l or "allgather" in l: return "comm (all-to-all / gather)"
    if "qk_int" in l or "attn" in l or "flash" in l or "transposepad" in l or "meanscale" in l or "quantint8" in l: return "attention (+sage quant)"
    if "scaled_mm" in l or "gemm" in l or "cutlass" in l or "nvjet" in l or "xmma" in l: return "gemm"
    if "memcpy" in l or "copy" in l or "cat" in l: return "copy"
    return "other"
agg = collections.defaultdict(float)
for e in P.key_averages():
    t = getattr(e, "self_device_time_total", 0)
    if t and e.device_type == torch.autograd.DeviceType.CUDA: agg[cat(e.key)] += t / 1e3
if R0:
    tot = sum(agg.values()); print("[sp_bench] world %d sp %s: wall %.1f ms (median of 6), CUDA %.1f ms: %s" % (dist.get_world_size(), a.sp, 1000 * np.median(ts), tot,
          ", ".join("%s %.1f ms" % kv for kv in sorted(agg.items(), key=lambda kv: -kv[1]))), flush=True)
dist.barrier(); dist.destroy_process_group()
