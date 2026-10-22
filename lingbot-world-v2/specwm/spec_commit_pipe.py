"""Commit offload (lever 1): two cloud GPUs, 14B only. rank 0 = denoiser (K uniform steps per chunk, never runs the clean commit pass),
rank 1 = committer (runs the clean t=0 pass on every finished chunk and ships that chunk's K/V ring slots back). Step 1 of chunk c+1 runs while
chunk c is being committed, on the denoiser's own last-step K/V of chunk c (overlap_probe: gap 0.29σ vs the 14B noise floor 0.32); the clean
K/V arrives before step 2. Per chunk the denoiser's critical path is K forwards instead of K+1.
usage: CUDA_VISIBLE_DEVICES=a python -m specwm.spec_commit_pipe --rank 0 --port P ... & CUDA_VISIBLE_DEVICES=b python -m specwm.spec_commit_pipe --rank 1 --port P ...
"""
import argparse, os, sys, json, time, torch, torch.distributed as dist
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--rank", type=int, required=True); ap.add_argument("--port", type=int, default=29800); ap.add_argument("--tasks", required=True); ap.add_argument("--seeds", required=True)
ap.add_argument("--steps", type=int, default=2); ap.add_argument("--out", required=True); ap.add_argument("--arm", default=""); ap.add_argument("--no-video", action="store_true"); ap.add_argument("--warm", type=int, default=1); ap.add_argument("--frames", type=int, default=161); ap.add_argument("--mode", default="uniform", help="uniform | chunk (chunk-wise adaptive with the 14B head)"); ap.add_argument("--chunk-frac", type=float, default=0.7); ap.add_argument("--eps", type=float, default=0.24); ap.add_argument("--max-area", type=int, default=480 * 832); ap.add_argument("--solo", action="store_true", help="single process: synchronous commits, no committer rank"); ap.add_argument("--stale", default="last", help="what step 1 of chunk c+1 sees in chunk c's slots: last = the denoiser's last-step K/V (noisy input), copy = chunk c-1's clean K/V copied forward"); ap.add_argument("--sync-first", type=int, default=0, help="commit the first N chunks on the denoiser itself (the image chunk 0 is special)")
a = ap.parse_args(); sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
dev = "cuda:0"; torch.cuda.set_device(0)
if a.solo: a.sync_first = 10 ** 6
else: os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(a.port)); dist.init_process_group("nccl", rank=a.rank, world_size=2)
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(B.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 64
for blk in B.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
K = a.steps; SIG = [float(s) for s in B.node_sigmas()]; arm = a.arm or (("commitpipe_u%d" % K if a.mode == "uniform" else "commitpipe_chunk%g" % a.chunk_frac) + ("_sync%d" % a.sync_first if a.sync_first else "") + ("_stalecopy" if a.stale == "copy" else "") + ("_area%d" % a.max_area if a.max_area != 480 * 832 else ""))
EE = None
if a.mode == "chunk" and a.rank == 0:
    from specwm.early_exit import EarlyExit; EE = EarlyExit(B, eps=a.eps, chunk_frac=a.chunk_frac)
def gen(seed, c): return torch.Generator().manual_seed(seed * 1009 + c * 7)
def slot(c):   # ring slot [off, off+S) of chunk c (fast_patch.sa_forward layout)
    fs = B._frame_seqlen; S_ = B._chunk_size * fs; size = B._caches["target"]["kv"][0]["k"].shape[1]; sink = B.target_model.blocks[0].self_attn.sink_size * fs
    sink_c = sink // S_; slots = (size - sink) // S_; off = c * S_ if c < sink_c + slots else sink + ((c - sink_c) % slots) * S_; return off, S_
def kv_buf():
    l0 = B._caches["target"]["kv"][0]["k"]; _, S_ = slot(0); return torch.empty(len(B._caches["target"]["kv"]), 2, S_, *l0.shape[2:], dtype=l0.dtype, device=dev)
def gather_kv(c, buf):
    off, S_ = slot(c)
    for l, L_ in enumerate(B._caches["target"]["kv"]): buf[l, 0].copy_(L_["k"][0, off:off + S_]); buf[l, 1].copy_(L_["v"][0, off:off + S_])
def scatter_kv(c, buf):
    off, S_ = slot(c)
    for l, L_ in enumerate(B._caches["target"]["kv"]): L_["k"][0, off:off + S_].copy_(buf[l, 0]); L_["v"][0, off:off + S_].copy_(buf[l, 1])
def sync(): torch.cuda.synchronize()
@torch.no_grad()
def run(task, seed, out):
    T = ALL[task]; t0 = time.time()
    cond = B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=a.frames, max_area=a.max_area); B.reset_caches(); N = cond["num_chunks"]; prep = time.time() - t0
    buf = [kv_buf(), kv_buf()]; chunks = []; steps_used = []; tm = dict(denoise=0.0, wait_kv=0.0, commit=0.0, send=0.0); req = None; tg = time.time()
    if a.rank == 0:
        for c in range(N):
            g = gen(seed, c); z = B.chunk_noise(c, g); t1 = time.time(); st = dict(t1=t1, pulled=False)
            def pull():   # clean K/V of chunk c-1 must be in place before step 2 (or before the next chunk if this one stops after step 1)
                if st["pulled"] or c <= a.sync_first: return
                sync(); tm["denoise"] += time.time() - st["t1"]; t2 = time.time(); req.wait(); scatter_kv(c - 1, buf[(c - 1) % 2]); sync(); tm["wait_kv"] += time.time() - t2; st["t1"] = time.time(); st["pulled"] = True
            if a.stale == "copy" and c > a.sync_first:   # stand-in for chunk c-1 (not yet committed): its predecessor's clean K/V
                o1, S_ = slot(c - 1); o0, _ = slot(c - 2)
                for L_ in B._caches["target"]["kv"]: L_["k"][0, o1:o1 + S_].copy_(L_["k"][0, o0:o0 + S_]); L_["v"][0, o1:o1 + S_].copy_(L_["v"][0, o0:o0 + S_])
            if EE is not None: x, ms, _ = EE.gen_chunk(c, z, g, sparse=False, after=lambda i: pull() if i == 0 else None); steps_used.append(ms)
            else:
                for i in range(K):
                    if i > 0: s_ = SIG[i]; z = (1 - s_) * x + s_ * B.randn_like(x, g)
                    x = B.target_x0(z, i, c).float()
                    if i == 0: pull()
                steps_used.append(K)
            pull(); sync(); tm["denoise"] += time.time() - st["t1"]
            chunks.append(x.cpu())
            if not a.solo: xs = x.to(torch.bfloat16).contiguous(); dist.send(xs, dst=1)
            if c < a.sync_first: t1 = time.time(); B.commit(x, c); sync(); tm["commit"] += time.time() - t1; req = None
            else: B._committed_frames += B._chunk_size; req = dist.irecv(buf[c % 2], src=1)   # no clean pass here
        if req is not None: req.wait()   # drain the last K/V
        gen_s = time.time() - tg; os.makedirs(out, exist_ok=True); lat = torch.cat(chunks, 1)
        torch.save(lat.half(), out + "/lat.pt"); dec = 0.0
        if not a.no_video: td = time.time(); B.decode_and_save([lat.to(dev)], out); dec = time.time() - td
        e2e = time.time() - t0; json.dump(dict(task=task, seed=seed, arm=arm, steps=K, chunks=N, prepare_s=round(prep, 2), gen_s=round(gen_s, 2), decode_s=round(dec, 2), e2e_s=round(e2e, 2), mean_steps=float(sum(steps_used) / max(1, len(steps_used))), times={k: round(v, 2) for k, v in tm.items()}), open(out + "/loop.json", "w"), indent=1)
        print("[COMMIT_PIPE] %s s%d %s gen %.1fs e2e %.1fs | denoise %.1f wait_kv %.1f steps %.2f" % (task, seed, arm, gen_s, e2e, tm["denoise"], tm["wait_kv"], sum(steps_used) / max(1, len(steps_used))), flush=True)
    else:
        for c in range(N):
            x = torch.empty(B.chunk_noise(c, gen(seed, c)).shape, dtype=torch.bfloat16, device=dev); dist.recv(x, src=0); t1 = time.time()
            B.commit(x.float(), c); sync(); tm["commit"] += time.time() - t1; t2 = time.time()
            if c >= a.sync_first: gather_kv(c, buf[c % 2]); dist.send(buf[c % 2], dst=0); sync(); tm["send"] += time.time() - t2
        print("[COMMIT_PIPE rank1] %s s%d commit %.1f send %.1f" % (task, seed, tm["commit"], tm["send"]), flush=True)
tasks = a.tasks.split(","); seeds = [int(s) for s in a.seeds.split(",")]
if a.warm: run(tasks[0], seeds[0], a.out + "/%s/%s_warm" % (tasks[0], arm))
for task in tasks:
    for seed in seeds: run(task, seed, a.out + "/%s/%s_s%d" % (task, arm, seed))
if not a.solo: dist.barrier()
print("[ALL_DONE]", flush=True)
