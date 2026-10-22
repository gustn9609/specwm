"""Upper bound of commit+verify fusion: time one 14B forward over two chunks (6 latent frames, 9360 tokens) vs two forwards over one chunk each,
same KV state (4 committed chunks, no eviction), FP8 + compile + SageAttention. usage: CUDA_VISIBLE_DEVICES=g python -m specwm.bench_fuse"""
import os, sys, json, time, numpy as np, torch
S = "/home/hk28965/specwm"; sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
T = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}["easy_urban_rain"]; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 500))
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage")
FP.fp8_swap(B.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 64
for blk in B.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); B.reset_caches(); sync = torch.cuda.synchronize
g = torch.Generator().manual_seed(0); K = 4
with torch.no_grad():
    for ci in range(K): x = B.chunk_noise(ci, g); B.commit(B.target_x0(x, 0, ci), ci)
M = B.target_model; kv = B._caches["target"]["kv"]; SN = [(l["global_end_index"].clone(), l["local_end_index"].clone()) for l in kv]
def restore():
    for l, (ge, le) in zip(kv, SN): l["global_end_index"].copy_(ge); l["local_end_index"].copy_(le)
def fwd(frames, ci):
    """one model forward over `frames` latent frames starting at chunk ci (t = 0)"""
    x = torch.randn(16, frames, *B._chunk_shape[2:], device="cuda:0"); kw = B._kwargs("target", ci)
    n = frames // 3; kw["y"] = [torch.cat(B._y_chunks[ci:ci + n], 1)]; kw["dit_cond_dict"] = {"c2ws_plucker_emb": torch.cat(B._emb_chunks[ci:ci + n], 2).chunk(1, dim=0)}
    kw["seq_len"] = B._seq_len * n
    with torch.amp.autocast("cuda", dtype=torch.bfloat16): return M(x=[x.to(torch.bfloat16)], t=torch.zeros(1, device="cuda:0"), cross_attn_first_call=False, **kw)[0]
def tm(f, n=6):
    with torch.no_grad():
        for _ in range(3): restore(); f()
        ts = []
        for _ in range(n): restore(); sync(); t = time.time(); f(); sync(); ts.append(time.time() - t)
    return 1000 * float(np.median(ts))
one_a = tm(lambda: fwd(3, K)); one_b = tm(lambda: (fwd(3, K), fwd(3, K + 1))); two = tm(lambda: fwd(6, K))
print("[fuse] 1 chunk %.1f ms | 2 chunks sequential %.1f ms | 2 chunks in one forward %.1f ms | saving %.1f %%" % (one_a, one_b, two, 100 * (1 - two / one_b)), flush=True)
