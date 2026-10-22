"""Equivalence check of fast_patch FLAGS ring_frame (frame-granular K/V ring, 10/13) against the chunk ring: same process, same compile
mode, same noise; chunks 0..N-1 of one scene generated (a) through the chunk path (B.target_x0 / B.commit, ring_frame off) and (b) through
per-frame-range calls (ring_frame on, f0 = 3c, m = 3). Reports per-chunk max / mean abs difference of x0 and within-eps. Then (c) runs a
variable-unit sequence (3,1,1,1,3,...) under ring_frame to make sure it produces finite, in-range latents.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.ring_frame_check [--chunks 7] [--dynamic 1]"""
import argparse, os, sys, json, numpy as np, torch
S = "/home/hk28965/specwm"; ap = argparse.ArgumentParser(); ap.add_argument("--chunks", type=int, default=7); ap.add_argument("--dynamic", type=int, default=1); ap.add_argument("--task", default="easy_lake"); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"; dev = "cuda:0"
B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True, ring_frame=False); FP.fp8_swap(B.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 512; _dyn.config.accumulated_recompile_limit = 8192
for blk in B.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=bool(a.dynamic))
B.pipe.text_encoder.model.to(dev); _te = B.pipe.text_encoder
class _TE:
    def __init__(s, te): s.te = te; s.model = te.model
    def __call__(s, texts, device): return s.te(texts, dev)
B.pipe.text_encoder = _TE(_te); sig = [float(s) for s in B.node_sigmas()]; T = ALL[a.task]
def gen(seed, c, k=0): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k)
def patch(x):
    c, f, h, w = x.shape; return x.view(c, f, 1, h // 2, 2, w // 2, 2).permute(1, 3, 5, 2, 4, 6, 0).reshape(f * (h // 2) * (w // 2), 64)
def call(z, t_val, f0, m):
    fs = B._frame_seqlen; C = B._caches["target"]; emb = torch.cat([E[f] for f in range(f0, f0 + m)], 2)
    kw = dict(context=[B._ctx[0]], seq_len=m * fs, y=[Y[:, f0:f0 + m]], dit_cond_dict={"c2ws_plucker_emb": emb.chunk(1, dim=0)}, kv_cache=C["kv"], crossattn_cache=C["cross"], current_start=f0 * fs, max_attention_size=B._kv_size, frame_seqlen=fs)
    with torch.amp.autocast("cuda", dtype=B.param_dtype): out = B.target_model(x=[z.to(B.param_dtype)], t=torch.full((1,), float(t_val), device=dev), cross_attn_first_call=not B._cross_init["target"], **kw)[0]
    B._cross_init["target"] = True; return out.float()
def den_frames(z, f0, m, g):
    for i in range(4):
        x = z - sig[i] * call(z, B._timesteps[i], f0, m)
        if i < 3: z = (1 - sig[i + 1]) * x + sig[i + 1] * B.randn_like(x, g)
    return x
def den_chunk(z, c, g):
    for i in range(4):
        x = B.target_x0(z, i, c).float()
        if i < 3: z = (1 - sig[i + 1]) * x + sig[i + 1] * B.randn_like(x, g)
    return x
with torch.no_grad():
    cond = B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); N = min(a.chunks, cond["num_chunks"]); Y = torch.cat(list(B._y_chunks), 1); E = [e for ch in B._emb_chunks for e in ch.split(1, dim=2)]
    B.reset_caches(); XA = []
    for c in range(N): x = den_chunk(B.chunk_noise(c, gen(7, c)), c, gen(7, c, 1)); B.commit(x, c); XA.append(x)
    B.reset_caches(); FP.FLAGS["ring_frame"] = True; XB = []
    for c in range(N): x = den_frames(B.chunk_noise(c, gen(7, c)), 3 * c, 3, gen(7, c, 1)); call(x, 0.0, 3 * c, 3); XB.append(x)
    print("| chunk | max abs diff | mean abs diff | within eps 0.24 |")
    for c in range(N):
        d = (XA[c] - XB[c]).abs(); tT = patch(XA[c]); sg = float(tT.norm(dim=1).median()); w = float((((patch(XB[c]) - tT).norm(dim=1) / sg) <= 0.24).float().mean())
        print("| %d | %.4f | %.5f | %.1f %% |" % (c, float(d.max()), float(d.mean()), 100 * w), flush=True)
    B.reset_caches(); f = 0; units = []; XC = []
    while f < 3 * N:
        m = 3 if (len(units) % 5 == 0) else 1; m = min(m, 3 * N - f); z = torch.randn((16, m, B._chunk_shape[2], B._chunk_shape[3]), generator=gen(7, 100 + f)).to(dev)
        x = den_frames(z, f, m, gen(7, 100 + f, 1)); call(x, 0.0, f, m); units.append((f, m)); XC.append(x); f += m
    xc = torch.cat(XC, 1); print("variable units", units); print("finite", bool(torch.isfinite(xc).all()), "abs mean %.3f max %.2f (fixed-chunk run: abs mean %.3f max %.2f)" % (float(xc.abs().mean()), float(xc.abs().max()), float(torch.cat(XA, 1).abs().mean()), float(torch.cat(XA, 1).abs().max())))
    # the frame-level run with the SAME actions (prepare's stream) but different unit boundaries: token distance to the fixed-chunk run per latent frame (content drifts, expected; a bug would show as garbage magnitudes or NaN)
    xa = torch.cat(XA, 1); print("per-frame within eps vs fixed-chunk run (different noise, so ~ the other-noise floor 29 %):", [round(float((((patch(xc[:, i:i + 1]) - patch(xa[:, i:i + 1])).norm(dim=1) / float(patch(xa[:, i:i + 1]).norm(dim=1).median())) <= 0.24).float().mean()), 2) for i in range(0, 3 * N, 3)])
print("[RFC_DONE]")
