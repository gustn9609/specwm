"""[10/13, lever 4 feasibility] Can the Fast 14B (trained with 3-latent-frame chunks = 0.75 s) run with 1- or 2-latent-frame chunks
(0.25 / 0.5 s), which is the only route to the 750 ms action-to-frame target? Stock 14B, 4 steps, deployed stack (FP8 / Sage / compile /
ring KV, sink 6 frames, window 18 frames), same seed; per scene one rollout per chunk size, timed per chunk after warm-up, decoded to mp4
for the VLM judge. Output <out>/<task>_s<seed>/cs<k>/spec.mp4 and <out>/<task>_s<seed>/stats.json (per chunk size: s per chunk, s per
video second, latent frames).  usage: CUDA_VISIBLE_DEVICES=g python -m specwm.chunk1_probe --out DIR [--cs 1,2,3] [--tasks a,b] [--seeds 7]"""
import argparse, os, sys, json, time, numpy as np, torch
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--cs", default="1,2,3"); ap.add_argument("--seeds", default="7")
ap.add_argument("--tasks", default=open(S + "/lingbot-world-v2/specwm_out/distill_heldout12.txt").read().strip()); ap.add_argument("--frames", type=int, default=161); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
tasks = a.tasks.split(","); seeds = [int(s) for s in a.seeds.split(",")]; CS = [int(x) for x in a.cs.split(",")]; dev = "cuda:0"
B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(B.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 512; _dyn.config.accumulated_recompile_limit = 8192
for blk in B.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
os.makedirs(a.out, exist_ok=True)
def gen(seed, c, k=0): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k)
sg = [float(s) for s in B.node_sigmas()]
for task in tasks:
    for seed in seeds:
        od = "%s/%s_s%d" % (a.out, task, seed)
        if os.path.exists(od + "/stats.json"): continue
        T = ALL[task]; R = dict(task=task, seed=seed, cs={}); os.makedirs(od, exist_ok=True)
        try:
            for cs in CS:
                with torch.no_grad():
                    cond = B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=a.frames, max_area=480 * 832, chunk_size=cs); B.reset_caches()
                    N = cond["num_chunks"]; X = []; tc = []
                    for c in range(N):
                        torch.cuda.synchronize(); t0 = time.time(); z = B.chunk_noise(c, gen(seed, c)); g = gen(seed, c, 1)
                        for i in range(len(sg)):
                            x = B.target_x0(z, i, c).float()
                            if i < len(sg) - 1: z = (1 - sg[i + 1]) * x + sg[i + 1] * B.randn_like(x, g)
                        B.commit(x, c); X.append(x); torch.cuda.synchronize(); tc.append(time.time() - t0)
                    lat = torch.cat(X, 1); os.makedirs(od + "/cs%d" % cs, exist_ok=True); B.decode_and_save([lat.to(dev)], od + "/cs%d" % cs)
                steady = tc[2:] if len(tc) > 4 else tc; vid_s = cs * 4 / 16.0   # seconds of video per chunk
                R["cs"][cs] = dict(chunks=N, lat_frames=int(lat.shape[1]), s_per_chunk=round(float(np.mean(steady)), 3), s_per_video_s=round(float(np.mean(steady)) / vid_s, 3), first_chunk_s=round(tc[0], 3))
                print("[CP] %s s%d cs%d: %d chunks, %.3f s/chunk (%.2fx real time), first %.2f s" % (task, seed, cs, N, R["cs"][cs]["s_per_chunk"], 1 / R["cs"][cs]["s_per_video_s"], tc[0]), flush=True)
            json.dump(R, open(od + "/stats.json", "w"))
        except Exception as e:
            import traceback; traceback.print_exc(); print("[CP_FAIL] %s s%d %r" % (task, seed, e), flush=True)
        finally:
            torch.cuda.empty_cache()
print("[CP_DONE]", flush=True)
