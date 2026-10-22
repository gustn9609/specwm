"""Anchored vs own-history draft gap per chunk (meeting 10/12 items B + D3).
One 14B 4-step rollout per (task, seed) (FP8 / Sage / compile / ring KV, window 18, 832x480, 161 frames = 13 chunks). At every chunk c the same
noise z0 is denoised by (i) the 14B (4 steps; steps 1-3 kept as the self-speculation reference), (ii) the 14B from a DIFFERENT noise (sample
diversity reference), (iii) the 1.3B ANCHORED on the teacher's history (its KV written from the 14B's committed latents, as gap_dump_cross did on
10/6), (iv) the 1.3B on its OWN history (a second KV set written from its own outputs = the student rollout). Per chunk and per latent frame we
store the token gap ||x0 - x0_T^(4)|| / sigma_T (sigma_T = median token norm of the 14B output; token = 1x2x2 latent patch) as mean and share <= eps.
At c = 0 both 1.3B variants are identical (empty KV): the gap there is the capacity gap with NO history, which answers Guandao's KV-fill question
directly. Both the teacher and the own-history student clips are decoded to mp4 so the per-chunk departure can be correlated with the VLM judge.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.anchor_gap --shard i/n --out DIR [--seeds 7,1234,4321] [--tasks a,b] [--eps 0.24]"""
import argparse, os, sys, json, time, numpy as np, torch
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--shard", default="0/1"); ap.add_argument("--out", required=True); ap.add_argument("--seeds", default="7,1234,4321")
ap.add_argument("--tasks", default=""); ap.add_argument("--eps", type=float, default=0.24); ap.add_argument("--video", type=int, default=1); ap.add_argument("--frames", type=int, default=161)
ap.add_argument("--d-lora", default="", help="draft LoRA (draft_distill4.py draft_lora*.pt) merged into the 1.3B weights before the run"); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
B43 = ("easy_lake,balloon_block_cut,bicycle_tire_pump,billiard_break,bowling_inertia,domino_falling,drawbridge_rising,drinking_fountain_button,expanding_foam_paper_cup_burst,"
       "expanding_sponge_water,gas_pump_nozzle_trigger,hydraulic_press_can_crush_10,hydraulic_press_can_crush,inflating_balloon_09,inflating_balloon,lightbulb_and_switch,"
       "lightbulb_and_switch_opposite,newtons_cradle,paper_over_spill,pedestrians_enter_crosswalk,plastic_bottle_hand_crush,pouring_water_into_cup,railroad_crossing_gates,"
       "seesaw_balance,shower_knob_turn,shower_knob_turn_opposite,spray_foam,sprinkler_valve_control,sprinkler_valve_control_opposite,sprint_start_gun,string_pull_lamp,"
       "string_pull_lamp_opposite,syringe_hydraulic,tape_retract,tea_dilution,tire_inflate,traffic_light_green,upipe_equilibrium,wet_pan_stove_on,easy_greatwall,"
       "easy_stonehenge_sunset,easy_urban_rain,easy_dragon_night")
tasks = (a.tasks or B43).split(","); si, sn = (int(x) for x in a.shard.split("/")); tasks = tasks[si::sn]; seeds = [int(s) for s in a.seeds.split(",")]
NODES = (0, 250, 500, 750)
Tg = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=NODES)
D = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=NODES, cfg_name="i2v-1.3B", assets_dir=FAST)
if a.d_lora:   # merge W += B A (block order q,k,v,o[,ffn0,ffn2]), as noise_calib.py / spec_pipe.py
    st = torch.load(a.d_lora, map_location="cpu"); it = iter(zip(st["A"], st["B"]))
    with torch.no_grad():
        for blk in D.target_model.blocks:
            for lin in [getattr(blk.self_attn, n) for n in ("q", "k", "v", "o")] + ([blk.ffn[0], blk.ffn[2]] if st["ffn"] else []): A_, B_ = next(it); lin.weight += (B_.float() @ A_.float()).to(lin.weight.device, lin.weight.dtype)
    print("[AG] merged draft LoRA", a.d_lora, "step", st.get("step"), flush=True)
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(Tg.target_model, True, keep=False)   # 1.3B stays bf16 as on 10/6
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 512; _dyn.config.accumulated_recompile_limit = 8192
for m in (Tg.target_model, D.target_model):
    for blk in m.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
os.makedirs(a.out, exist_ok=True); dev = "cuda:0"
def patch(x):
    c, f, h, w = x.shape; return x.view(c, f, 1, h // 2, 2, w // 2, 2).permute(1, 3, 5, 2, 4, 6, 0).reshape(f * (h // 2) * (w // 2), 64)
def gen(seed, c, k=0): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k)
def run4(Bd, z, c, g):   # 4-step rollout of one chunk on binding Bd from noise z; returns the x0 estimate after each step
    xs = []; sg = [float(s) for s in Bd.node_sigmas()]
    for i in range(len(sg)):
        x = Bd.target_x0(z, i, c).float(); xs.append(x)
        if i < len(sg) - 1: z = (1 - sg[i + 1]) * x + sg[i + 1] * Bd.randn_like(x, g)
    return xs
def use(cs, fl): D._caches = cs; D._cross_init = fl
def stat(x, tT, sg, f_):   # per-token gap -> mean, share <= eps, per latent frame
    gp = ((patch(x) - tT).norm(dim=1) / sg).view(f_, -1)
    return dict(mean=round(float(gp.mean()), 4), frac=round(float((gp <= a.eps).float().mean()), 4), fmean=[round(float(v), 4) for v in gp.mean(1)], ffrac=[round(float(v), 4) for v in (gp <= a.eps).float().mean(1)])
done = 0
for task in tasks:
    for seed in seeds:
        od = "%s/%s_s%d" % (a.out, task, seed)
        if os.path.exists(od + "/stats.json"): continue
        T = ALL[task]; t0 = time.time(); os.makedirs(od, exist_ok=True)
        try:
            with torch.no_grad():
                cond = Tg.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=a.frames, max_area=480 * 832); Tg.reset_caches()
                D.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=a.frames, max_area=480 * 832); D.reset_caches()
                cA, fA = D._caches, D._cross_init                                     # anchored: KV written from the teacher's latents
                cO, fO = {"target": D._mk_cache()}, {"draft": False, "target": False}  # own history: KV written from the 1.3B's own outputs
                N = cond["num_chunks"]; R = dict(task=task, seed=seed, eps=a.eps, chunks=[]); XT, XO = [], []
                for c in range(N):
                    g = gen(seed, c); z0 = Tg.chunk_noise(c, g)
                    xsT = run4(Tg, z0, c, g); xT = xsT[-1]                                   # (i) teacher
                    xsT2 = run4(Tg, Tg.chunk_noise(c, gen(seed, c, 2)), c, gen(seed, c, 3)) # (ii) teacher, other noise
                    use(cA, fA); xsA = run4(D, z0, c, gen(seed, c, 1))                     # (iii) 1.3B anchored on teacher history
                    use(cO, fO); xsO = run4(D, z0, c, gen(seed, c, 1))                     # (iv) 1.3B on its own history (student rollout)
                    f_ = xT.shape[1]; tT = patch(xT); sg = float(tT.norm(dim=1).median())
                    r = dict(c=c, sigma=round(sg, 4), T=[stat(x, tT, sg, f_) for x in xsT[:3]], T2=stat(xsT2[-1], tT, sg, f_), A=[stat(x, tT, sg, f_) for x in xsA], O=stat(xsO[-1], tT, sg, f_),
                             AO=stat(xsO[-1], patch(xsA[-1]), sg, f_))                      # AO: own-history draft vs anchored draft = the history's effect alone
                    R["chunks"].append(r)
                    Tg.commit(xT, c); use(cA, fA); D.commit(xT, c); use(cO, fO); D.commit(xsO[-1], c); XT.append(xT); XO.append(xsO[-1])
                    print("[chunk] %s s%d c%02d  T1 %.3f/%.0f%%  T2 %.3f/%.0f%%  A4 %.3f/%.0f%%  O4 %.3f/%.0f%%" % (task, seed, c, r["T"][0]["mean"], 100 * r["T"][0]["frac"], r["T2"]["mean"], 100 * r["T2"]["frac"], r["A"][-1]["mean"], 100 * r["A"][-1]["frac"], r["O"]["mean"], 100 * r["O"]["frac"]), flush=True)
                R["gen_s"] = round(time.time() - t0, 1)
                if a.video:
                    for nm, X in (("teacher", XT), ("student", XO)):
                        os.makedirs(od + "/" + nm, exist_ok=True); Tg.decode_and_save([torch.cat(X, 1).to(dev)], od + "/" + nm)
                R["total_s"] = round(time.time() - t0, 1); json.dump(R, open(od + "/stats.json", "w"))
            done += 1; print("[AG] %s s%d %.0fs (%d)" % (task, seed, time.time() - t0, done), flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print("[AG_FAIL] %s s%d %r" % (task, seed, e), flush=True)
        finally:
            D._caches = {}; torch.cuda.empty_cache()
print("[AG_DONE] shard %s" % a.shard, flush=True)
