"""Single-forward timing of speed variants on a warmed KV cache (chunk K after K commits). Variants: patched (fast_patch flags on), +compile, +fp8, +fp8+compile.
Also: output rel. diff of one forward vs the patched bf16 forward (same input). usage: CUDA_VISIBLE_DEVICES=g python -m specwm.bench_kern --model 14b"""
import argparse, os, sys, json, time, copy, numpy as np, torch
S = "/home/hk28965/specwm"; ap = argparse.ArgumentParser(); ap.add_argument("--model", default="14b"); ap.add_argument("--K", type=int, default=5); ap.add_argument("--out", default=""); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
T = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}["sand_filling_up_jar"]; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750)) if a.model == "14b" else \
    V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), cfg_name="i2v-1.3B", assets_dir=FAST)
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True)
B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); B.reset_caches(); sync = torch.cuda.synchronize
g = torch.Generator().manual_seed(0)
with torch.no_grad():
    for ci in range(a.K): x = B.chunk_noise(ci, g); B.commit(B.target_x0(x, 0, ci), ci)
xin = B.chunk_noise(a.K, g); M = B.target_model; R = {}
def snap():   # K chunks fit in the KV window (no eviction): the measured forward only writes the next slot, so restoring the indices is enough
    return [(l["global_end_index"].clone(), l["local_end_index"].clone()) for l in B._caches["target"]["kv"]]
def restore(sn):
    for l, (ge, le) in zip(B._caches["target"]["kv"], sn): l["global_end_index"].copy_(ge); l["local_end_index"].copy_(le)
SN = snap(); ref = None
def run(name, n=5):
    global ref
    with torch.no_grad():
        ts = []
        for i in range(n + 3):
            restore(SN); sync(); t = time.time(); y = B.target_x0(xin, 0, a.K); sync(); ts.append(time.time() - t)
    if ref is None: ref = y
    R[name] = dict(ms=round(1000 * float(np.median(ts[3:])), 1), first_ms=round(1000 * ts[0], 1), rel=round(float((y - ref).norm() / ref.norm()), 5))
    print("[kern]", name, R[name], flush=True)
run("patched")
FP.fp8_swap(M, True); run("fp8"); FP.fp8_swap(M, False)
for blk in M.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
run("compile")
FP.fp8_swap(M, True); run("fp8+compile")
json.dump(R, open(a.out or "../exp/kveval/bench_kern_%s.json" % a.model, "w"), indent=1); print("[KERN_DONE]")
