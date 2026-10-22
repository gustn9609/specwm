"""Distillation data for the draft: 14B target-only rollouts in the deployed configuration (2 steps, FP8, compile, SageAttention, ring KV) with
spec_pipe's per-chunk noise (seed*1009 + c*7 + {node0: 0, T: 2}), saving per clip:  lat.pt [16, F, h, w] fp16 (final chunks = teacher-forcing context)
and x0.pt {node: [N, 16, 3, h, w] fp16} = the 14B's x0 prediction at every node of every chunk. The noise is not stored (re-drawn from the seed).
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.target_dump --tasks @list --seeds 1,2 --out ROOT"""
import argparse, os, sys, json, time, torch
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--tasks", required=True); ap.add_argument("--seeds", default="1,2,3,4"); ap.add_argument("--out", required=True)
ap.add_argument("--t-steps", default="0,500"); ap.add_argument("--area", type=int, default=480 * 832, help="[10/16] 704x384 = 270336"); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
TI = tuple(int(x) for x in a.t_steps.split(","))
B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=TI)
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(B.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 64; _dyn.config.accumulated_recompile_limit = 4096
for blk in B.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
B.pipe.text_encoder.model.to("cuda:0"); _te = B.pipe.text_encoder
class _TE:
    def __init__(s, te): s.te = te; s.model = te.model
    def __call__(s, texts, device): return s.te(texts, torch.device("cuda:0"))
B.pipe.text_encoder = _TE(_te)
def gen(seed, c, path): return torch.Generator().manual_seed(seed * 1009 + c * 7 + {"node0": 0, "D": 1, "T": 2}[path])
tasks = open(a.tasks[1:]).read().split() if a.tasks.startswith("@") else a.tasks.split(",")
for task in tasks:
    T = ALL[task]
    for seed in [int(s) for s in a.seeds.split(",")]:
        out = "%s/%s_s%d" % (a.out, task, seed)
        if os.path.exists(out + "/x0.pt"): continue
        t0 = time.time()
        with torch.no_grad():
            cond = B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=a.area); B.reset_caches()
            N = cond["num_chunks"]; sig = [float(s) for s in B.node_sigmas()]; X0 = {i: [] for i in range(len(sig))}; lat = []
            for c in range(N):
                z = B.chunk_noise(c, gen(seed, c, "node0")); g = gen(seed, c, "T"); x = B.target_x0(z, 0, c); X0[0].append(x.half().cpu())
                for i in range(1, len(sig)):
                    s_ = sig[i]; z = (1 - s_) * x + s_ * B.randn_like(x, g); x = B.target_x0(z, i, c); X0[i].append(x.half().cpu())
                x = x.float(); B.commit(x, c); lat.append(x.half().cpu())
        os.makedirs(out, exist_ok=True); torch.save(torch.cat(lat, 1), out + "/lat.pt")
        torch.save({i: torch.stack(v) for i, v in X0.items()}, out + "/x0.pt"); json.dump(dict(task=task, seed=seed, steps=list(TI), sigmas=sig, chunks=N), open(out + "/meta.json", "w"))
        print("[DUMP] %s s%d %.1fs" % (task, seed, time.time() - t0), flush=True)
print("[DUMP_DONE]", flush=True)
