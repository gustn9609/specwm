"""Future-state impact of one approximated chunk (feedback 10/7, 5.2). From a dense 14B prefix (chunks < c0), branch at chunk c0:
  A  : dense 4-step chunk c0                                   (reference branch)
  A' : dense 4-step chunk c0 with a different re-noise sequence (noise-only floor: how much the future changes anyway)
  B1 : oracle-mask early exit with dense recompute (approximation error only, eps 0.24)
  B2 : learned-mask early exit with sparse K/V reuse (the deployed sampler)
  B3 : uniform 2-step chunk c0
then every branch continues with 3 dense 4-step chunks (same noise for all branches). Measures per future chunk f = 1..3: latent token gap
between branch and A (sigma units) and DINOv2 CLS cosine similarity of decoded frames to A; also the gap of the intervened chunk itself.
If B's future divergence exceeds the A' floor by more than its local error predicts, local gap is not the whole story.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.branch_probe --tasks a,b --seeds 4321 --c0 3,7 --out DIR"""
import argparse, os, sys, json, time, numpy as np, torch
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--tasks", required=True); ap.add_argument("--seeds", default="4321"); ap.add_argument("--c0", default="3,7"); ap.add_argument("--future", type=int, default=3); ap.add_argument("--out", required=True); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
from specwm.early_exit import EarlyExit, patch
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(B.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 64
for blk in B.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=True)
EE = EarlyExit(B, eps=0.24); os.makedirs(a.out, exist_ok=True); dev = "cuda:0"; SIG = [float(s) for s in B.node_sigmas()]
def gen(seed, c, k=0): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k)
@torch.no_grad()
def dense(c, z, g, K=4):
    x = None
    for i in range(K):
        x = B.target_x0(z, i, c).float()
        if i < K - 1: s_ = SIG[i + 1]; z = (1 - s_) * x + s_ * B.randn_like(x, g)
    return x
def snapshot():   # deep copy of the KV / cross caches (the branch point)
    return {k: dict(kv=[{kk: vv.to("cpu", copy=True) for kk, vv in l.items()} for l in v["kv"]], cross=[{kk: (vv.to("cpu", copy=True) if torch.is_tensor(vv) else vv) for kk, vv in l.items()} for l in v["cross"]]) for k, v in B._caches.items()}, dict(B._cross_init), B._committed_frames   # CPU copy (the 14B window cache is ~30 GB)
def restore(snap):
    caches, ci, cf = snap
    for k, v in caches.items():
        for l, l2 in zip(B._caches[k]["kv"], v["kv"]):
            for kk in l: l[kk].copy_(l2[kk])
        for l, l2 in zip(B._caches[k]["cross"], v["cross"]):
            for kk in l:
                if torch.is_tensor(l[kk]): l[kk].copy_(l2[kk])
    B._cross_init = dict(ci); B._committed_frames = cf
# DINO
from transformers import AutoImageProcessor
from transformers.models.dinov2.modeling_dinov2 import Dinov2Model
proc = AutoImageProcessor.from_pretrained(S + "/hf/dinov2-large"); dino = Dinov2Model.from_pretrained(S + "/hf/dinov2-large").to(dev).eval().half()
@torch.no_grad()
def dino_feat(lat):   # [16, F, H, W] latent -> per-frame CLS (decoded with the VAE)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16): im = B.pipe.vae.decode([lat.to(dev)])[0].float().clamp(-1, 1)
    fr = ((im + 1) * 127.5).permute(1, 2, 3, 0).byte().cpu().numpy(); out = []
    for i in range(0, len(fr), 32):
        x = proc(images=list(fr[i:i + 32]), return_tensors="pt")["pixel_values"].to(dev).half(); out.append(torch.nn.functional.normalize(dino(pixel_values=x).last_hidden_state[:, 0].float(), dim=-1).cpu())
    return torch.cat(out)
recs = []
for task in a.tasks.split(","):
    T = ALL[task]
    for seed in [int(s) for s in a.seeds.split(",")]:
        for c0 in [int(x) for x in a.c0.split(",")]:
            with torch.no_grad():
                cond = B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); B.reset_caches(); N = cond["num_chunks"]
                for c in range(c0): B.commit(dense(c, B.chunk_noise(c, gen(seed, c)), gen(seed, c, 1)), c)   # shared dense prefix
                snap = snapshot(); z0 = B.chunk_noise(c0, gen(seed, c0)); branches = {}
                def run_branch(name, make_chunk):
                    restore(snap); x = make_chunk(); B.commit(x, c0); fut = []
                    for f in range(1, a.future + 1):
                        c = c0 + f
                        if c >= N: break
                        xf = dense(c, B.chunk_noise(c, gen(seed, c)), gen(seed, c, 1)); B.commit(xf, c); fut.append(xf)
                    branches[name] = (x, fut)
                run_branch("A", lambda: dense(c0, z0, gen(seed, c0, 1)))
                run_branch("A_prime", lambda: dense(c0, z0, gen(seed, c0, 21)))
                xsA = None
                def b1():
                    g = gen(seed, c0, 1); st = g.get_state(); xs = EE.dense_trajectory(c0, z0, g); masks = EE.oracle_masks(xs); g.set_state(st); return EE.gen_chunk(c0, z0, g, masks=masks, sparse=False)[0]
                run_branch("B1_oracle_dense", b1)
                run_branch("B2_learned_sparse", lambda: EE.gen_chunk(c0, z0, gen(seed, c0, 1))[0])
                run_branch("B3_uniform2", lambda: dense(c0, z0, gen(seed, c0, 1), K=2))
                # measure
                xA, futA = branches["A"]; tA = patch(xA); sgA = float(tA.norm(dim=1).median()); featA = [dino_feat(f) for f in futA]
                r = dict(task=task, seed=seed, c0=c0, branches={})
                for name, (x, fut) in branches.items():
                    if name == "A": continue
                    row = dict(local_gap=float(((patch(x) - tA).norm(dim=1) / sgA).mean()), future_gap=[], future_dino=[])
                    for f, (xf, xfA) in enumerate(zip(fut, futA)):
                        tf = patch(xfA); row["future_gap"].append(float(((patch(xf) - tf).norm(dim=1) / float(tf.norm(dim=1).median())).mean()))
                        row["future_dino"].append(float((dino_feat(xf) * featA[f]).sum(1).mean()))
                    r["branches"][name] = row
                recs.append(r); print("[BRANCH] %s s%d c0=%d " % (task, seed, c0) + " | ".join("%s: local %.3f future %s" % (n, v["local_gap"], ["%.2f" % g for g in v["future_gap"]]) for n, v in r["branches"].items()), flush=True)
                json.dump(recs, open(a.out + "/records.json", "w"))
names = ["A_prime", "B1_oracle_dense", "B2_learned_sparse", "B3_uniform2"]
print("| branch at c0 | local gap of chunk c0 ↓ | future gap f=1 / 2 / 3 (σ) ↓ | future DINO cos to A f=1 / 2 / 3 ↑ |"); print("|---|---|---|---|")
for n in names:
    L = [r["branches"][n] for r in recs if n in r["branches"]]
    print("| %s | %.3f | %s | %s |" % (n, np.mean([x["local_gap"] for x in L]), " / ".join("%.2f" % np.mean([x["future_gap"][f] for x in L if len(x["future_gap"]) > f]) for f in range(a.future)), " / ".join("%.2f" % np.mean([x["future_dino"][f] for x in L if len(x["future_dino"]) > f]) for f in range(a.future))))
print("[BRANCH_DONE]", flush=True)
