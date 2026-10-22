"""Where does a 14B step go? Time one denoising forward of the 14B (chunk 4 of jar, dense context) with all tokens vs an active subset (75/50/25 %),
and the KV commit forward. usage: CUDA_VISIBLE_DEVICES=g python -m specwm.step_bench"""
import os, sys, json, time, torch
S = "/home/hk28965/specwm"; sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(B.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 64
for blk in B.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=True)
T = ALL["sand_filling_up_jar"]; g = torch.Generator().manual_seed(1)
with torch.no_grad():
    cond = B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); B.reset_caches()
    for c in range(4):
        z = B.chunk_noise(c, g); x = B.target_x0(z, 0, c).float(); B.commit(x, c)
    c = 4; z = B.chunk_noise(c, g); x0 = B.target_x0(z, 0, c).float(); L = x0.shape[1] * (x0.shape[2] // 2) * (x0.shape[3] // 2)
    def tm(f, n=5):
        f(); torch.cuda.synchronize(); t = time.time()
        for _ in range(n): f()
        torch.cuda.synchronize(); return (time.time() - t) / n
    rows = []
    for attn in ("sage", "flash"):
        FP.FLAGS["attn"] = attn; FP.ACTIVE["a"] = None; rows.append(("dense step, %s attention" % attn, 1.0, tm(lambda: B.target_x0(z, 1, c))))
    FP.FLAGS["attn"] = "sage"
    for frac in (0.75, 0.5, 0.25, 0.1):
        n = int(L * frac); idx = torch.randperm(L, device="cuda")[:n].sort().values; pad = (-n) % 128; idx = torch.cat([idx, idx[-1:].expand(pad)]) if pad else idx
        FP.ACTIVE["a"] = dict(idx=idx, n=n, L=L); rows.append(("sparse step, %d %% of tokens active (flash)" % int(100 * frac), frac, tm(lambda: B.target_x0(z, 1, c))))
    FP.ACTIVE["a"] = None; rows.append(("KV commit (clean pass, all tokens)", 1.0, tm(lambda: B.commit(x0, c))))
print("| forward | active tokens | time per chunk ↓ | relative to dense sage |"); print("|---|---|---|---|")
for n_, f, t in rows: print("| %s | %.0f %% | %.3f s | %.2f |" % (n_, 100 * f, t, t / rows[0][2]))
print("[BENCH_DONE]")
