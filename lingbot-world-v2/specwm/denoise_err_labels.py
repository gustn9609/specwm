"""Experiment 1: is a model's own denoising error an absolute drift detector? On the human-labelled clips (exp/drift_lat/*.pt: 13 chunk latents,
clip-level severe/drift labels, arm, seed) compute per chunk c >= 1, with the clip's OWN previous chunks committed as context (teacher forcing):
  e_T(sigma) = ||x0_T((1-s) x + s eps) - x|| / ||x||   for the 14B (sigmas --sigmas), and e_D(sigma) for the 1.3B draft,
plus the shared-noise 14B 'anchor' distance d1 = ||x0_T^(step1 from z0) - x|| / sigma_tok (how far the chunk is from what the 14B would draw).
Output <out>/records.json (one row per clip: labels + per-chunk scores) and a clip-level AUC table (max / mean over chunks) vs 'severe'.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.denoise_err_labels --out DIR --shard i/n [--max-clips N] [--split test]"""
import argparse, os, sys, json, glob, time, numpy as np, torch
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--shard", default="0/1"); ap.add_argument("--sigmas", default="0.3,0.5,0.7")
ap.add_argument("--max-clips", type=int, default=0); ap.add_argument("--split", default="test", help="test = the drift critic's held-out seeds/arms (+ sand/castle tasks); all = everything")
ap.add_argument("--no-draft", action="store_true"); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
Tg = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
D = None if a.no_draft else V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), cfg_name="i2v-1.3B", assets_dir=FAST)
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(Tg.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 64
for blk in Tg.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
SIG = [float(x) for x in a.sigmas.split(",")]; os.makedirs(a.out, exist_ok=True); dev = "cuda:0"
TEST_SEEDS = {109, 110, 111, 112, 113, 114, 115, 116, 9999}; TEST_ARMS = {"none", "target", "periodic_2", "e3_0.4541", "e3_0.5387", "critic_main_-8.6183", "critic_main_-8.7456", "ir_-0.7", "ir_-2.2", "never1p3b"}
files = sorted(glob.glob(S + "/exp/drift_lat/*__s*.pt")); meta = []
for f in files:
    b = os.path.basename(f)[:-3]; task, arm, sd = b.split("__")[0], b.split("__")[1], int(b.split("__s")[-1])
    if task not in ALL: continue
    is_test = (sd in TEST_SEEDS) or (arm in TEST_ARMS) or task in ("sand_filling_up_jar", "sandcastle_wave_impact")
    if a.split == "test" and not is_test: continue
    meta.append((f, task, arm, sd))
import random, hashlib
meta.sort(key=lambda m: hashlib.sha1(m[0].encode()).hexdigest())   # deterministic shuffle
if a.max_clips:   # balanced: half severe, half not, 3-rater batches first
    lab = {m[0]: torch.load(m[0], map_location="cpu") for m in meta}; three = lambda m: lab[m[0]].get("n_raters", 1) == 3
    pos = [m for m in meta if lab[m[0]]["severe"]]; neg = [m for m in meta if not lab[m[0]]["severe"]]
    pos.sort(key=lambda m: not three(m)); neg.sort(key=lambda m: not three(m)); meta = pos[:a.max_clips // 2] + neg[:a.max_clips // 2]
si, sn = (int(x) for x in a.shard.split("/")); meta = meta[si::sn]
print("clips in this shard: %d" % len(meta), flush=True)
def patch(x):
    c, f, h, w = x.shape; return x.view(c, f, 1, h // 2, 2, w // 2, 2).permute(1, 3, 5, 2, 4, 6, 0).reshape(f * (h // 2) * (w // 2), 64)
recs = []; done_tasks = {}; t_start = time.time()
for n, (f, task, arm, sd) in enumerate(meta):
    d = torch.load(f, map_location="cpu"); lat = d["lat"].float().to(dev); N = lat.shape[1] // 3; T = ALL[task]
    try:
        with torch.no_grad():
            if done_tasks.get("task") != task:   # prepare once per task (text / image / action conditioning)
                cond = Tg.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832)
                if D is not None: D.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832)
                done_tasks["task"] = task
            Tg.reset_caches();
            if D is not None: D.reset_caches()
            if lat.shape[2:] != tuple(Tg._chunk_shape[2:]): print("[SKIP shape] %s %s" % (f, tuple(lat.shape)), flush=True); continue
            rows = []
            for c in range(N):
                x = lat[:, 3 * c:3 * c + 3]
                if c >= 1:
                    g = torch.Generator().manual_seed(1000 + c); eps = Tg.randn_like(x, g); r = dict(chunk=c)
                    for s in SIG:
                        z = (1 - s) * x + s * eps
                        xt = Tg.target_x0(z, 0, c) if False else (z - s * Tg.target_flow(z, s, c)).float(); r["eT_%.1f" % s] = float(((xt - x) ** 2).mean().sqrt() / (x ** 2).mean().sqrt())
                        if D is not None: xd = (z - s * D.target_flow(z, s, c)).float(); r["eD_%.1f" % s] = float(((xd - x) ** 2).mean().sqrt() / (x ** 2).mean().sqrt())
                    z0 = Tg.chunk_noise(c, torch.Generator().manual_seed(7 * c + 3)); x1 = Tg.target_x0(z0, 0, c).float(); tx = patch(x); sg = float(tx.norm(dim=1).median())
                    r["d1"] = float(((patch(x1) - tx).norm(dim=1) / sg).mean()); rows.append(r)
                Tg.commit(x, c)
                if D is not None: D.commit(x, c)
            recs.append(dict(file=os.path.basename(f), task=task, arm=arm, seed=sd, severe=bool(d["severe"]), drift=bool(d["drift"]), batch=d.get("batch"), cats=d.get("cats", []), rows=rows))
    except Exception as e: print("[FAIL] %s %r" % (f, e), flush=True)
    if (n + 1) % 10 == 0: print("[ERR] %d/%d clips, %.0fs" % (n + 1, len(meta), time.time() - t_start), flush=True); json.dump(recs, open("%s/records_%d.json" % (a.out, si), "w"))
json.dump(recs, open("%s/records_%d.json" % (a.out, si), "w"))
from sklearn.metrics import roc_auc_score
y = [r["severe"] for r in recs]
if len(set(y)) > 1:
    keys = sorted({k for r in recs for row in r["rows"] for k in row if k != "chunk"}); print("| score | clip AUC (max over chunks) ↑ | clip AUC (mean) ↑ |"); print("|---|---|---|")
    for k in keys: print("| %s | %.3f | %.3f |" % (k, roc_auc_score(y, [max(row[k] for row in r["rows"]) for r in recs]), roc_auc_score(y, [np.mean([row[k] for row in r["rows"]]) for r in recs])))
print("[ERR_DONE] shard %s, %d clips" % (a.shard, len(recs)), flush=True)
