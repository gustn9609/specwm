"""[10/16 stage 3 prep] per-block sensitivity of the causal-fast 14B: teacher-forced on bench clips (a704_43/teacher/<task>/lat.pt as history),
for chunk c and node i the model's x0 with block b bypassed (identity) vs the intact model; error = ||x0_b - x0||^2 / ||x0||^2. Input at node 0 =
the chunk noise, at node i > 0 = the teacher's final chunk re-noised to sigma_i (a stand-in for the step-(i-1) estimate). Output <out>/sens.json
{err: [task][chunk][node][block], mean_by_node_block, rank}. usage: CUDA_VISIBLE_DEVICES=g python -m specwm.block_sens --out DIR [--tasks a,b --chunks 4,8,12]"""
import argparse, os, sys, json, time, torch, numpy as np
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--src", default=S + "/lingbot-world-v2/specwm_out/twomach/a704_43/teacher")
ap.add_argument("--tasks", default="easy_lake,domino_falling,pouring_water_into_cup,newtons_cradle,billiard_break,easy_urban_rain,traffic_light_green,balloon_block_cut"); ap.add_argument("--chunks", default="4,8,12")
ap.add_argument("--area", type=int, default=270336); ap.add_argument("--seed", type=int, default=1234); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"; dev = "cuda:0"
B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750)); m = B.target_model
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(m, True, keep=False)
sig = [float(s) for s in B.node_sigmas()]; NB = len(m.blocks); FWD = [blk.forward for blk in m.blocks]
def log(*x): print("[sens %s]" % time.strftime("%H:%M:%S"), *x, flush=True)
def gen(c, k): return torch.Generator().manual_seed(a.seed * 1009 + c * 7 + k * 100003)
ERR = {}; CH = [int(c) for c in a.chunks.split(",")]
with torch.no_grad():
    for task in a.tasks.split(","):
        T = ALL[task]; L = torch.load("%s/%s/lat.pt" % (a.src, task)).float().to(dev); cond = B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=a.area); B.reset_caches(); N = cond["num_chunks"]
        assert tuple(L.shape[2:]) == tuple(B._chunk_shape[2:]), (L.shape, B._chunk_shape); ERR[task] = {}
        for c in range(N):
            if c in CH:
                xT = L[:, 3 * c:3 * c + 3]; ERR[task][c] = {}
                for i in range(len(sig)):
                    g = gen(c, i); z = B.chunk_noise(c, g) if i == 0 else (1 - sig[i]) * xT + sig[i] * B.randn_like(xT, g)
                    ref = B.target_x0(z, i, c).float(); e = []
                    for b in range(NB):
                        m.blocks[b].forward = (lambda x, **kw: x); xb = B.target_x0(z, i, c).float(); m.blocks[b].forward = FWD[b]
                        e.append(float(((xb - ref) ** 2).mean() / (ref ** 2).mean()))
                    ERR[task][c][i] = e; log(task, "chunk", c, "node", i, "min/median/max block err %.3f / %.3f / %.3f" % (min(e), float(np.median(e)), max(e)), "worst blocks", list(np.argsort(e)[-4:][::-1]))
            B.commit(L[:, 3 * c:3 * c + 3], c)
        json.dump(dict(err=ERR, sigmas=sig, tasks=a.tasks, chunks=CH), open(a.out + "/sens.json", "w"))
M = np.zeros((len(sig), NB)); n = 0
for t in ERR.values():
    for c in t.values():
        for i, e in c.items(): M[int(i)] += np.array(e)
    n += len(t)
M /= max(n, 1); tot = M.mean(0); rank = [int(b) for b in np.argsort(tot)]   # least sensitive first
json.dump(dict(err=ERR, sigmas=sig, tasks=a.tasks, chunks=CH, mean_by_node_block=M.tolist(), mean_block=tot.tolist(), rank_least_sensitive_first=rank), open(a.out + "/sens.json", "w"))
log("mean err per block (x100):", [round(100 * v, 1) for v in tot]); log("least sensitive first:", rank); print("[SENS_DONE]", flush=True)
