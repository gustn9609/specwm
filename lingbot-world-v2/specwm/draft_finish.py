"""Draft-then-Finish: device 1.3B drafts each chunk at 832x480 (its K/V holds the cloud's finished chunks, lagging one chunk as in a real
pipeline), cloud 14B runs ONE finishing forward at 832: re-noise the draft to sigma_f and predict x0. --commit separate: an extra clean t=0
commit pass (2 cloud forwards / chunk, upper bound); --commit fused: the K/V the finishing forward wrote into the ring slot is kept as the
commit (1 cloud forward / chunk). Writes lat.pt (finished chunks), video.mp4 (full Wan VAE), taehv.mp4 (TAEHV decode), loop.json with
per-chunk cloud / device times. usage: CUDA_VISIBLE_DEVICES=g python -m specwm.draft_finish --tasks a,b --seeds 1234 --sigma-node 750
--commit fused --out ROOT [--draft-nodes 0,500] [--lag 1]"""
import argparse, os, sys, json, time, torch, torch.nn.functional as F, imageio, numpy as np
_reserve = torch.empty(int(os.environ.get("DF_RESERVE_GB", "80")) << 30, dtype=torch.uint8, device="cuda:0"); del _reserve   # hold the GPU in torch's cache before loading
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--tasks", required=True); ap.add_argument("--seeds", default="1234"); ap.add_argument("--out", required=True); ap.add_argument("--arm", default="")
ap.add_argument("--sigma-node", type=int, default=750, help="scheduler index of the 14B finishing node: 750 -> sigma 0.625 (its own last node), 850 -> 0.468, 930 -> 0.273")
ap.add_argument("--configs", default="", help="several arms in one process: fused:750,separate:850,...");
ap.add_argument("--commit", default="fused", choices=["fused", "separate"]); ap.add_argument("--draft-nodes", default="0,500", help="1.3B draft schedule (scheduler indices)")
ap.add_argument("--lag", type=int, default=1, help="1: the device drafts chunk c with finished chunks <= c-2 and its own draft of c-1 (cloud result of c-1 not back yet); 0: finished chunks <= c-1")
ap.add_argument("--warm", type=int, default=1); ap.add_argument("--no-video", action="store_true"); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); sys.path.insert(0, S + "/tools/taehv"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
from taehv import TAEHV
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"; dev = "cuda:0"
DN = [int(v) for v in a.draft_nodes.split(",")]
Tg = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 750, 850, 930))
D = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=tuple(DN), cfg_name="i2v-1.3B", assets_dir=FAST)
for B in (Tg, D):   # cache the prompt encoding (resolution-independent)
    enc = B.pipe.text_encoder; cache = {}
    class _C:
        def __init__(s, e, c): s.model = e.model; s.e = e; s.c = c
        def __call__(s, texts, device):
            k = (tuple(texts), str(device))
            if k not in s.c: s.c[k] = s.e(texts, device)
            return s.c[k]
    B.pipe.text_encoder = _C(enc, cache)
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(Tg.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 512; _dyn.config.accumulated_recompile_limit = 8192
for blk in Tg.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
for blk in D.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
TAE = TAEHV(S + "/tools/taehv/taew2_1.pth").to(dev).half().eval()
NODES = [0, 750, 850, 930]; SD = [float(s) for s in D.node_sigmas()]; sync = torch.cuda.synchronize
def gen(seed, c, k=0): return torch.Generator().manual_seed(seed * 1009 + c * 7 + 31 * k)
def fit(x, shape): return x if tuple(x.shape[2:]) == tuple(shape[2:]) else F.interpolate(x.permute(1, 0, 2, 3), size=tuple(shape[2:]), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
@torch.no_grad()
def run(task, seed, out, commit, node):
    K = NODES.index(node); SF = float(Tg.node_sigmas()[K]); arm = "df_%s_n%d_lag%d" % (commit, node, a.lag)
    T = ALL[task]; t0 = time.time()
    cond = Tg.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); Tg.reset_caches(); N = cond["num_chunks"]
    D.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); D.reset_caches(); prep = time.time() - t0
    finals = []; drafts = []; c_cloud = []; c_dev = []; tg = time.time()
    for c in range(N):
        sync(); t1 = time.time()
        if a.lag and c >= 2: D.commit(fit(finals[c - 2].to(dev), D._chunk_shape), c - 2)   # the finished chunk c-2 has arrived: overwrite the device's own draft K/V
        g = gen(seed, c); z = D.chunk_noise(c, g); x = None
        for j in range(len(DN)):
            if j > 0: s_ = SD[j]; z = (1 - s_) * x + s_ * D.randn_like(x, g)
            x = D.target_x0(z, j, c).float()
        xd = x; sync(); c_dev.append(time.time() - t1)
        t2 = time.time(); xd_t = fit(xd, Tg._chunk_shape); zf = (1 - SF) * xd_t + SF * Tg.randn_like(xd_t, gen(seed, c, 1))
        xf = Tg.target_x0(zf, K, c).float()   # the 14B's single finishing forward (writes its K/V into chunk c's ring slot)
        if commit == "separate": Tg.commit(xf, c)
        else: Tg._committed_frames += Tg._chunk_size   # fused: keep the K/V the finishing forward wrote
        sync(); c_cloud.append(time.time() - t2); finals.append(xf.half().cpu()); drafts.append(xd.half().cpu())
        t3 = time.time(); D.commit(fit(xd if a.lag else xf, D._chunk_shape), c); sync(); c_dev[-1] += time.time() - t3   # lag: device continues from its own draft until the final arrives
    gen_s = time.time() - tg; os.makedirs(out, exist_ok=True); lat = torch.cat(finals, 1); torch.save(lat, out + "/lat.pt")
    t4 = time.time(); v = TAE.decode_video(lat.to(dev).permute(1, 0, 2, 3)[None].half(), parallel=True, show_progress_bar=False)[0]; sync(); tae_s = time.time() - t4
    imageio.mimwrite(out + "/taehv.mp4", list((v.float().clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()), fps=16, quality=8)
    if not a.no_video: Tg.decode_and_save([lat.to(dev).float()], out); os.replace(out + "/spec.mp4", out + "/video.mp4") if os.path.exists(out + "/spec.mp4") else None
    cc = np.array(c_cloud[1:]); dd = np.array(c_dev[1:])
    R = dict(task=task, seed=seed, arm=arm, commit=commit, sigma_f=SF, draft_nodes=DN, lag=a.lag, chunks=N, prepare_s=round(prep, 2), gen_s=round(gen_s, 2), cloud_chunk_s=[round(x, 4) for x in c_cloud], device_chunk_s=[round(x, 4) for x in c_dev],
             cloud_med=round(float(np.median(cc)), 3), device_med=round(float(np.median(dd)), 3), taehv_s=round(tae_s, 2), pipelined_fps=round(12 / max(np.median(cc), np.median(dd) + tae_s / N), 1))
    json.dump(R, open(out + "/loop.json", "w"), indent=1); print("[DF] %s s%d %s cloud %.3f s/chunk device %.3f s/chunk -> pipelined %.1f fps (gen %.1f s serial)" % (task, seed, arm, R["cloud_med"], R["device_med"], R["pipelined_fps"], gen_s), flush=True)
tasks = a.tasks.split(","); seeds = [int(s) for s in a.seeds.split(",")]
CF = [(c.split(":")[0], int(c.split(":")[1])) for c in a.configs.split(",")] if a.configs else [(a.commit, a.sigma_node)]
for cm, nd in CF:
    arm = "df_%s_n%d_lag%d" % (cm, nd, a.lag)
    if a.warm: run(tasks[0], seeds[0], a.out + "/%s/%s_warm" % (tasks[0], arm), cm, nd); a.warm = 0
    for t in tasks:
        for s in seeds:
            o = a.out + "/%s/%s_s%d" % (t, arm, s)
            if not os.path.exists(o + "/loop.json"): run(t, s, o, cm, nd)
print("[ALL_DONE]", flush=True)
