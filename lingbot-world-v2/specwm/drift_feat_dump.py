"""14B hidden features for the human-labelled drift clips (exp/drift_lat): teacher-force each clip through the 14B commit pass (t = 0, the pass
every pipeline already runs) and keep, per chunk, the layer-20 and layer-30 block outputs pooled to a 3x5 grid + global mean (16 x 5120, fp16).
Writes exp/drift_feat/<clip>.pt = {task, arm, seed, severe, drift, cats, group, f20 [N,16,5120], f30 [N,16,5120]}.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.drift_feat_dump --shard i/n [--n-other 320 --n-jc 300]"""
import argparse, os, sys, glob, json, random, numpy as np, torch, torch.nn.functional as F
S = "/home/hk28965/specwm"; OUT = S + "/exp/drift_feat"; os.makedirs(OUT, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("--shard", default="0/1"); ap.add_argument("--n-other", type=int, default=320); ap.add_argument("--n-jc", type=int, default=300); ap.add_argument("--layers", default="20,30"); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 500))
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(B.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 64
for blk in B.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
LAY = [int(x) for x in a.layers.split(",")]; H = {}
def mk(k):
    def f(m, i, o): H[k] = (o[0] if isinstance(o, tuple) else o)
    return f
for L in LAY: B.target_model.blocks[L - 1].register_forward_hook(mk(L))
F_ = [f for f in sorted(glob.glob(S + "/exp/drift_lat/*.pt")) if "/cam_" not in f]; random.seed(0)
meta = [(f, torch.load(f, map_location="cpu", mmap=True)) for f in F_]; JC = ("sand_filling_up_jar", "sandcastle_wave_impact")
other = [m for m in meta if m[1]["task"] not in JC]; jc = [m for m in meta if m[1]["task"] in JC]; random.shuffle(other); random.shuffle(jc)
sel = other[:a.n_other] + jc[:a.n_jc]; si, sn = [int(v) for v in a.shard.split("/")]; sel = sel[si::sn]
sel.sort(key=lambda m: m[1]["task"])   # group by task: one prepare per task
def pool(h, f_, h2, w2):   # [1, L, C] tokens of a chunk -> [16, C]: global mean + 3x5 grid means
    x = h[0].float().view(f_, h2, w2, -1).permute(3, 0, 1, 2)   # [C, f, h, w]
    g = F.adaptive_avg_pool3d(x, (1, 3, 5)).reshape(x.shape[0], 15).t(); return torch.cat([x.mean((1, 2, 3))[None], g]).half().cpu()
cur = None; n = 0
for f, d in sel:
    dst = OUT + "/" + os.path.basename(f)
    if os.path.exists(dst): continue
    T = ALL.get(d["task"])
    if T is None: continue
    with torch.no_grad():
        if cur != d["task"]: B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); cur = d["task"]
        B.reset_caches(); L = d["lat"].float().to("cuda"); N = L.shape[1] // 3
        if tuple(L.shape[2:]) != tuple(B._chunk_shape[2:]): print("[feat] grid mismatch, skip", os.path.basename(f), flush=True); continue
        f_, h2, w2 = 3, L.shape[2] // 2, L.shape[3] // 2; feats = {k: [] for k in LAY}
        for c in range(N):
            B.commit(L[:, 3 * c:3 * c + 3], c)
            for k in LAY: feats[k].append(pool(H[k], f_, h2, w2))
    rec = dict(task=d["task"], arm=d["arm"], seed=int(d["seed"]), severe=bool(d["severe"]), drift=bool(d["drift"]), cats=list(d.get("cats", [])), group="jc" if d["task"] in JC else "other")
    for k in LAY: rec["f%d" % k] = torch.stack(feats[k])
    torch.save(rec, dst); n += 1
    if n % 20 == 0: print("[feat] %d done (%s)" % (n, d["task"]), flush=True)
print("[FEAT_DONE] %d" % n, flush=True)
