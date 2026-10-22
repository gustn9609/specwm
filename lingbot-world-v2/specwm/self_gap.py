"""[10/13, lever 1 eval] Does a node-0 self-distilled LoRA (specwm/self_distill14.py) bring the 14B's step-1 estimate closer to its own
4-step output on UNSEEN scenes, without blurring? Deployed stack (FP8 / Sage / compile / ring KV, window 18, 832x480, 161 frames).
Per (task, seed), phase A = one stock 14B 4-step rollout (teacher forcing of the stock history); at every chunk from the same z0:
  T1..T3 : the stock step-1..3 x0 estimates (self-speculation reference, as facts §21)
  L1     : the step-1 x0 estimate with the LoRA gated ON at node 0 only
  L4     : LoRA step 1 followed by the stock steps 2-4 with the SAME later noise as the stock run (does the changed first step hurt the full path?)
  gaps   : ||x - xT|| / sigma_T per 1x2x2 token (mean, share <= eps) vs the stock final xT; also L1 vs T1 (how far the LoRA moved step 1).
Phase B / C = 1-step-every-chunk rollouts on their OWN history (stock; LoRA) decoded to mp4 — the worst-case blur check for the VLM judge;
the stock 4-step teacher clip is decoded as the judge control.  Output <out>/<task>_s<seed>/{stats.json, teacher|stock1|lora1/spec.mp4}.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.self_gap --out DIR --lora specwm_out/selfd14/T3/lora_s1000.pt [--tasks a,b] [--seeds 7]"""
import argparse, os, sys, json, time, numpy as np, torch
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--lora", required=True); ap.add_argument("--seeds", default="7")
ap.add_argument("--tasks", default=open(S + "/lingbot-world-v2/specwm_out/distill_heldout12.txt").read().strip()); ap.add_argument("--eps", type=float, default=0.24)
ap.add_argument("--video", type=int, default=1); ap.add_argument("--frames", type=int, default=161); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
tasks = a.tasks.split(","); seeds = [int(s) for s in a.seeds.split(",")]; dev = "cuda:0"
B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(B.target_model, True, keep=False)
GATE = torch.zeros((), device=dev)
class LoRA(torch.nn.Module):   # gated additive LoRA around the (FP8) linear; GATE = 1 only for the node-0 forward
    def __init__(self, base, A, Bm): super().__init__(); self.base = base; self.A = A.to(dev, torch.bfloat16); self.Bt = Bm.t().contiguous().to(dev, torch.bfloat16)
    def forward(self, x): return self.base(x) + (GATE.to(x.dtype) * ((x.to(torch.bfloat16) @ self.A.t()) @ self.Bt)).to(x.dtype)
st = torch.load(a.lora, map_location="cpu"); it = iter(zip(st["A"], st["B"]))
for blk in B.target_model.blocks:
    for nm in ("q", "k", "v", "o"): A_, B_ = next(it); setattr(blk.self_attn, nm, LoRA(getattr(blk.self_attn, nm), A_, B_))
    if st["ffn"]:
        for i in (0, 2): A_, B_ = next(it); blk.ffn[i] = LoRA(blk.ffn[i], A_, B_)
print("[SG] wrapped node-0 LoRA", a.lora, "step", st.get("step"), "target", st.get("target"), flush=True)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 512; _dyn.config.accumulated_recompile_limit = 8192
for blk in B.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
os.makedirs(a.out, exist_ok=True)
def patch(x):
    c, f, h, w = x.shape; return x.view(c, f, 1, h // 2, 2, w // 2, 2).permute(1, 3, 5, 2, 4, 6, 0).reshape(f * (h // 2) * (w // 2), 64)
def gen(seed, c, k=0): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k)
def run4(z, c, g, lora_first):   # 4-step rollout from noise z; LoRA on at node 0 only if lora_first; returns x0 after each step
    xs = []; sg = [float(s) for s in B.node_sigmas()]
    for i in range(len(sg)):
        GATE.fill_(1.0 if (lora_first and i == 0) else 0.0); x = B.target_x0(z, i, c).float(); GATE.fill_(0.0); xs.append(x)
        if i < len(sg) - 1: z = (1 - sg[i + 1]) * x + sg[i + 1] * B.randn_like(x, g)
    return xs
def stat(x, tT, sg):
    gp = (patch(x) - tT).norm(dim=1) / sg; return dict(mean=round(float(gp.mean()), 4), frac=round(float((gp <= a.eps).float().mean()), 4))
def rollout1(T, seed, N, lora):   # 1-step-per-chunk rollout on its own history
    B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=a.frames, max_area=480 * 832); B.reset_caches(); X = []
    for c in range(N):
        GATE.fill_(1.0 if lora else 0.0); x = B.target_x0(B.chunk_noise(c, gen(seed, c)), 0, c).float(); GATE.fill_(0.0); B.commit(x, c); X.append(x)
    return X
done = 0
for task in tasks:
    for seed in seeds:
        od = "%s/%s_s%d" % (a.out, task, seed)
        if os.path.exists(od + "/stats.json"): continue
        T = ALL[task]; t0 = time.time(); os.makedirs(od, exist_ok=True)
        try:
            with torch.no_grad():
                cond = B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=a.frames, max_area=480 * 832); B.reset_caches()
                N = cond["num_chunks"]; R = dict(task=task, seed=seed, eps=a.eps, lora=a.lora, chunks=[]); XT = []
                for c in range(N):
                    z0 = B.chunk_noise(c, gen(seed, c)); xsT = run4(z0, c, gen(seed, c), False); xT = xsT[-1]; xsL = run4(z0, c, gen(seed, c), True)
                    tT = patch(xT); sg = float(tT.norm(dim=1).median())
                    r = dict(c=c, sigma=round(sg, 4), T=[stat(x, tT, sg) for x in xsT[:3]], L1=stat(xsL[0], tT, sg), L2=stat(xsL[1], tT, sg), L4=stat(xsL[3], tT, sg), L1_vs_T1=stat(xsL[0], patch(xsT[0]), sg))
                    R["chunks"].append(r); GATE.fill_(0.0); B.commit(xT, c); XT.append(xT)
                    print("[chunk] %s s%d c%02d  T1 %.3f/%.0f%%  T2 %.3f/%.0f%%  L1 %.3f/%.0f%%  L4 %.3f/%.0f%%  L1-T1 %.3f" % (task, seed, c, r["T"][0]["mean"], 100 * r["T"][0]["frac"], r["T"][1]["mean"], 100 * r["T"][1]["frac"], r["L1"]["mean"], 100 * r["L1"]["frac"], r["L4"]["mean"], 100 * r["L4"]["frac"], r["L1_vs_T1"]["mean"]), flush=True)
                R["gen_s"] = round(time.time() - t0, 1)
                if a.video:
                    clips = {"teacher": XT, "stock1": rollout1(T, seed, N, False), "lora1": rollout1(T, seed, N, True)}
                    for nm, X in clips.items(): os.makedirs(od + "/" + nm, exist_ok=True); B.decode_and_save([torch.cat(X, 1).to(dev)], od + "/" + nm)
                R["total_s"] = round(time.time() - t0, 1); json.dump(R, open(od + "/stats.json", "w"))
            done += 1; print("[SG] %s s%d %.0fs (%d)" % (task, seed, time.time() - t0, done), flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print("[SG_FAIL] %s s%d %r" % (task, seed, e), flush=True)
        finally:
            torch.cuda.empty_cache()
print("[SG_DONE]", flush=True)
