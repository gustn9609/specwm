"""Re-issue the main comparison tables on two axes: blind severe-clean (3-rater majority of each batch) x camera following (VGGT relative rotation vs
the commanded poses, mean error over 39 frames; exp/geo). Also 'clean AND following' = clean with mean camera error <= --follow-deg.
usage: clean_follow_tables.py [--follow-deg 50] > paper/overnight/clean_follow_tables.md"""
import os, glob, json, argparse, collections, numpy as np
S = "/home/hk28965/specwm"; ap = argparse.ArgumentParser(); ap.add_argument("--follow-deg", type=float, default=50.0); a = ap.parse_args()
SEV = {"teleport", "color_collapse", "texture_collapse", "freeze"}
GEO = {}
for f in glob.glob(S + "/exp/geo/*.json"):
    g = json.load(open(f)); GEO[(g["task"], g["arm"], g["seed"])] = float(np.mean(g["cam_err"]))
def batch(name):
    d = S + "/paper/overnight/" + name; mp = json.load(open(d + "/mapping.json"))
    R = {r: {x["id"]: x for f in sorted(glob.glob("%s/r%d*.json" % (d, r))) for x in json.load(open(f))} for r in (1, 2, 3)}
    out = {}
    for i, v in mp.items():
        if not all(i in R[r] for r in (1, 2, 3)): continue
        sev = sum(R[r][i]["drift"] and bool(set(R[r][i].get("categories", [])) & SEV) for r in (1, 2, 3)) >= 2
        out[(v["task"], v["arm"], v["seed"])] = not sev
    return out
def table(title, name, arms, labels, note=""):
    L = batch(name); print("\n### %s\n%s\n" % (title, note))
    print("| arm | n | severe-clean ↑ | mean camera error ↓ (deg) | clean AND following (err ≤ %d°) ↑ | no geo |" % a.follow_deg); print("|---|---|---|---|---|---|")
    rows = []
    for arm, lab in zip(arms, labels):
        ks = [k for k in L if k[1] == arm]; cl = [L[k] for k in ks]; ce = [GEO[k] for k in ks if k in GEO]; both = [L[k] and GEO[k] <= a.follow_deg for k in ks if k in GEO]
        rows.append((lab, len(ks), sum(cl), np.mean(ce) if ce else float("nan"), sum(both), len(ks) - len(ce)))
    for lab, n, c, e, b, miss in rows: print("| %s | %d | %d/%d (%.2f) | %.1f | %d/%d (%.2f) | %d |" % (lab, n, c, n, c / max(n, 1), e, b, n, b / max(n, 1), miss))
table("Fresh seeds 101–116 (pre-registered batch, 32 clips/arm)", "blind_prereg",
      ["base70", "plh_plan6_ck400_l4c_st0_never", "plh_plan6_ck400_cr_st0_never", "sdvg_t-2p2_ir", "never"],
      ["Base-70 (v1)", "v1 plan (l4c) + render = v2-fast-CFG", "v1 plan (cr) + render = v2-safe", "SDVG port τ −2.2", "Fast alone"])
table("Planner variants, 22 base cases (one batch)", "blind_planvar2",
      ["plhv2_plan6_ck400_l4_st0_never", "plhv2_plan6_ck400_g15_st0_never", "plhv2_plan6_ck400_g2s1_st0_never", "plhv2_plan6_ck400_l4c_st0_never"],
      ["v2 bid plan l4 (no CFG) + render", "v2 plan g15 + render", "v2 plan g2s1 + render", "v2 plan l4c + render"])
table("Held-out seeds 109–116 (one batch)", "blind_planconf", ["plhv2_plan6_ck400_l4_st0_never", "plhv2_plan6_ck400_l4c_st0_never"], ["v2 plan l4 + render", "v2 plan l4c + render"])
table("Plan + verify (fresh seeds, separate batch)", "blind_pqplan2", ["pqplan_t00005_r3"], ["v1 plan + render + p/q verify + redraw"])
table("1.3B→14B speculative loop, 22 base cases (one batch)", "blind_loop",
      ["none", "target", "periodic_2", "e3_0.5387", "critic_main_-8.7456", "ir_-2.2", "ir_-0.7"],
      ["1.3B only", "14B only", "periodic:2", "③ τ0.54", "generator critic τ−8.75", "ImageReward τ−2.2 (SDVG)", "ImageReward τ−0.7 (SDVG paper)"])
table("Action-conditioned AR alone (38 cases)", "blind_fast1p3b", ["never1p3b", "never"], ["LingBot v2 1.3B causal-fast", "LingBot v2 14B causal-fast"])
