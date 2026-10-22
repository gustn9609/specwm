"""Per-arm time of the plan->render speed-up variants (plan gen + render gen + SR), mean over available cases. usage: python rfast_timing.py"""
import os, json, glob, numpy as np
O = "/home/hk28965/specwm/lingbot-world-v2/specwm_out"; V0 = O + "/spec_v0"; SR = O + "/rfast_sr"
SEEDS = [1234, 4321, 9999] + list(range(101, 117)); TASKS = ("sand_filling_up_jar", "sandcastle_wave_impact")
J = lambda p: json.load(open(p)) if os.path.exists(p) else {}
ARMS = [("plan 576x304 + render 4 nodes (current)", "planv2_l4", "plhv2_plan6_ck400_l4_st0_never", None),
        ("plan + render 3 nodes", "planv2_l4", "plhv2_plan6_ck400_l4_n023_never", None),
        ("plan + render 2 nodes", "planv2_l4", "plhv2_plan6_ck400_l4_n03_never", None),
        ("plan + render 576x304 + SR", "planv2_l4", "plhv2_plan6_ck400_l4_a576_never", "plhv2_plan6_ck400_l4_a576sr_never"),
        ("plan + render 448x256 + SR", "planv2_l4", "plhv2_plan6_ck400_l4_a448_never", "plhv2_plan6_ck400_l4_a448sr_never"),
        ("plan 448x256 + render 4 nodes", "planv2_l4a114688", "plhv2_plan6_ck400_l4a114688_st0_never", None),
        ("plan + SR only (no render)", "planv2_l4", None, "plansr_l4")]
out = {}
for lab, plan, rend, sr in ARMS:
    P, R, Q = [], [], []
    for t in TASKS:
        for s in SEEDS:
            pj = J("%s/%s/%s_s%d/plan.json" % (V0, t, plan, s)); rj = J("%s/%s/%s_s%d/spec_v0.json" % (V0, t, rend, s)) if rend else {"gen_s": 0.0}
            sj = J("%s/%s/%s_s%d/sr_timing.json" % (SR, t, sr, s)) if sr else {"sr_s": 0.0}
            if "gen_s" in pj and "gen_s" in rj and "sr_s" in sj: P.append(pj["gen_s"]); R.append(rj["gen_s"]); Q.append(sj["sr_s"])
    if P: out[lab] = dict(n=len(P), plan=float(np.mean(P)), render=float(np.mean(R)), sr=float(np.mean(Q)), total=float(np.mean(P) + np.mean(R) + np.mean(Q)))
    print("%-42s n=%2d plan %5.1f render %5.1f SR %4.1f total %6.1f" % ((lab,) + ((out[lab]["n"], out[lab]["plan"], out[lab]["render"], out[lab]["sr"], out[lab]["total"]) if lab in out else (0, 0, 0, 0, 0))))
json.dump(out, open("/home/hk28965/specwm/exp/kveval/rfast_timing.json", "w"), indent=1)
