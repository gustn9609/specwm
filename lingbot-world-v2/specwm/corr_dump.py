"""[10/16 stage 2 data] corrector pairs built the Correction-Training way (corrupt the good sample with the model's own flawed process):
for every 4-node teacher clip (target_dump n4_704: lat.pt), the 576x320 model is anchored on the teacher history (teacher chunks bicubic-
downsampled and committed) and chunk c's teacher latent is downsampled, re-noised to sigma_500 (0.833) and denoised with ONE step at 576
= a low-res draft that keeps the teacher chunk's coarse layout; it is upsampled to the teacher grid and paired with the cloud's node-750
re-noise draw. Output <out>/<clip>/pairs.pt {pairs: [{c, xd (upsampled), eps}], ...}; target = lat.pt chunk c (in the trainer).
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.corr_dump --data specwm_out/kd_data/n4_704 --out specwm_out/kd_data/corr_576 [--draft-area 184320]"""
import argparse, os, sys, glob, json, time, torch
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--draft-area", type=int, default=184320); ap.add_argument("--seed-k", type=int, default=1)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
TT = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"; dev = "cuda:0"
B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 500)); m = B.target_model
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(m, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 64; _dyn.config.accumulated_recompile_limit = 4096
for blk in m.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
sig = [float(s) for s in B.node_sigmas()]; S1 = sig[1]; S3 = 0.6246873736381531   # node-750 sigma of the 4-node schedule (shift 5)
def rs(x, hw): return torch.nn.functional.interpolate(x.permute(1, 0, 2, 3), size=tuple(hw), mode="bicubic", align_corners=False).permute(1, 0, 2, 3).contiguous()
def gen(seed, c, k): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k * 100003)
for d in sorted(glob.glob(a.data + "/*_s*")):
    if not os.path.exists(d + "/lat.pt"): continue
    name = os.path.basename(d); out = a.out + "/" + name
    if os.path.exists(out + "/pairs.pt"): continue
    meta = json.load(open(d + "/meta.json")); T = TT[meta["task"]]; seed = meta["seed"]; L = torch.load(d + "/lat.pt").float().to(dev); N = meta["chunks"]; t0 = time.time()
    with torch.no_grad():
        B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=a.draft_area); B.reset_caches(); hw = tuple(B._chunk_shape[2:]); HW = tuple(L.shape[2:]); pairs = []
        for c in range(N):
            xT = L[:, 3 * c:3 * c + 3]; xs = rs(xT, hw)
            if c > 0:
                g = gen(seed, c, a.seed_k); z = (1 - S1) * xs + S1 * B.randn_like(xs, g); xd = B.target_x0(z, 1, c).float(); xd_up = rs(xd.half().float(), HW)
                eps = torch.randn(xT.shape, generator=gen(seed, c, a.seed_k + 1)).to(dev); pairs.append(dict(c=c, xd=xd_up.half().cpu(), eps=eps.half().cpu()))
            B.commit(xs, c)
    os.makedirs(out, exist_ok=True); torch.save(dict(pairs=pairs, task=meta["task"], seed=seed, draft_area=a.draft_area, sigma_draft=S1, sigma_corr=S3), out + "/pairs.pt")
    print("[CORR_DUMP] %s %d pairs %.1fs" % (name, len(pairs), time.time() - t0), flush=True)
print("[CORR_DUMP_DONE]", flush=True)
