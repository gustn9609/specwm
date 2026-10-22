"""Strategy 2 (cloud layout step, device detail steps) and velocity guidance, measured with the token gap to the 14B 4-step output.
14B target-only rollout (4-step, chunks committed to both caches). At each chunk c >= 1 with the same z0:
  A4     = 14B final (reference);   A1 = 14B step-1 estimate (the 'anchor')
  draft  = 1.3B 4 steps from z0 (its own re-noise)                                  -> plain draft
  hmode  = 1.3B steps 2..4 started from the 14B's step-1 estimate A1 (re-noised with the 14B's schedule)   -> cloud layout, device detail
  hmode2 = same but the 1.3B runs only its last 2 steps (sigma 0.83, 0.63) from A1                        -> cheaper device part
  guide_w = step 1 velocity v = v_D + w (v_T - v_D) (w = 0.5, 1.0), then the 1.3B finishes steps 2..4       -> velocity guidance
Reports mean token gap to A4 and the share of tokens within eps 0.24 for each, plus the 14B's own step-1 gap and different-noise variability.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.anchor_probe --tasks a,b --seeds 4321,9999 --out DIR"""
import argparse, os, sys, json, time, numpy as np, torch
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--tasks", required=True); ap.add_argument("--seeds", default="4321"); ap.add_argument("--out", required=True); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
Tg = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
D = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), cfg_name="i2v-1.3B", assets_dir=FAST)
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(Tg.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 64
for blk in Tg.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
for blk in D.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
os.makedirs(a.out, exist_ok=True); SIG = [float(s) for s in Tg.node_sigmas()]
def patch(x):
    c, f, h, w = x.shape; return x.view(c, f, 1, h // 2, 2, w // 2, 2).permute(1, 3, 5, 2, 4, 6, 0).reshape(f * (h // 2) * (w // 2), 64)
def gen(seed, c, k): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k)
@torch.no_grad()
def finish(B, x, c, g, start):   # continue from clean estimate x at node `start` (re-noise to node start+1 .. end)
    for i in range(start + 1, len(SIG)): s_ = SIG[i]; z = (1 - s_) * x + s_ * B.randn_like(x, g); x = B.target_x0(z, i, c).float()
    return x
recs = []
for task in a.tasks.split(","):
    T = ALL[task]
    for seed in [int(s) for s in a.seeds.split(",")]:
        with torch.no_grad():
            cond = Tg.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); Tg.reset_caches()
            D.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); D.reset_caches(); N = cond["num_chunks"]
            for c in range(N):
                z0 = Tg.chunk_noise(c, gen(seed, c, 0)); A1 = Tg.target_x0(z0, 0, c).float(); A4 = finish(Tg, A1, c, gen(seed, c, 1), 0)
                if c >= 1:
                    t4 = patch(A4); sg = float(t4.norm(dim=1).median()); G = lambda x: ((patch(x) - t4).norm(dim=1) / sg)
                    out = {}
                    vT = Tg.target_flow(z0, SIG[0], c).float(); vD = D.target_flow(z0, SIG[0], c).float(); xD1 = (z0 - SIG[0] * vD).float()
                    out["draft"] = finish(D, xD1, c, gen(seed, c, 2), 0)
                    out["hmode"] = finish(D, A1, c, gen(seed, c, 3), 0)
                    out["hmode2"] = finish(D, A1, c, gen(seed, c, 4), 1)              # device runs only the last 2 steps from the anchor
                    out["hmode1"] = finish(D, A1, c, gen(seed, c, 6), 2)              # device runs only the last step (sigma 0.63) from the anchor
                    for w in (0.5, 1.0): xg = (z0 - SIG[0] * (vD + w * (vT - vD))).float(); out["guide%g" % w] = finish(D, xg, c, gen(seed, c, 5), 0)
                    out["anchor_only"] = A1
                    zB = Tg.chunk_noise(c, gen(seed, c, 11)); out["T_other_noise"] = finish(Tg, Tg.target_x0(zB, 0, c).float(), c, gen(seed, c, 12), 0)
                    out["T_hmode_14B"] = finish(Tg, A1, c, gen(seed, c, 13), 0)        # 14B itself finishing from its own anchor with a different re-noise (schedule-noise floor)
                    r = dict(task=task, seed=seed, chunk=c)
                    for k, x in out.items(): g = G(x); r[k] = dict(gap=float(g.mean()), within=float((g <= 0.24).float().mean()))
                    recs.append(r)
                Tg.commit(A4, c); D.commit(A4, c)
        print("[ANCHOR] %s s%d" % (task, seed), flush=True); json.dump(recs, open(a.out + "/records.json", "w"))
print("| proposal for the 14B's 4-step output | mean token gap ↓ (σ) | tokens within ε 0.24 ↑ | device cost (1.3B steps) | cloud cost (14B forwards) |"); print("|---|---|---|---|---|")
COST = {"draft": ("4", "0"), "hmode": ("3", "1"), "hmode2": ("2", "1"), "hmode1": ("1", "1"), "guide0.5": ("4", "1"), "guide1": ("4", "1"), "anchor_only": ("0", "1"), "T_other_noise": ("0", "4 (different noise)"), "T_hmode_14B": ("0", "4 (own anchor, other re-noise)")}
for k in ("draft", "guide0.5", "guide1", "hmode", "hmode2", "hmode1", "anchor_only", "T_hmode_14B", "T_other_noise"):
    print("| %s | %.3f | %.0f %% | %s | %s |" % (k, np.mean([r[k]["gap"] for r in recs]), 100 * np.mean([r[k]["within"] for r in recs]), *COST[k]))
print("[ANCHOR_DONE]", flush=True)
