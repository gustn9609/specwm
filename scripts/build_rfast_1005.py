"""paper/1005/plan_render_speedup: per (scene, seed) side-by-side of the plan->render speed-up variants + README (time + blind clean counts)."""
import os, json, subprocess, glob, collections
S = "/home/hk28965/specwm"; O = S + "/lingbot-world-v2/specwm_out"; V0 = O + "/spec_v0"; SR = O + "/rfast_sr"; P = S + "/paper/1005/plan_render_speedup"; os.makedirs(P, exist_ok=True)
T = json.load(open(S + "/exp/kveval/rfast_timing.json"))
ARMS = [("current 4 nodes", "plan 576x304 + render 4 nodes (current)", V0, "plhv2_plan6_ck400_l4_st0_never"),
        ("3 nodes", "plan + render 3 nodes", V0, "plhv2_plan6_ck400_l4_n023_never"),
        ("2 nodes", "plan + render 2 nodes", V0, "plhv2_plan6_ck400_l4_n03_never"),
        ("render 576+SR", "plan + render 576x304 + SR", SR, "plhv2_plan6_ck400_l4_a576sr_never"),
        ("render 448+SR", "plan + render 448x256 + SR", SR, "plhv2_plan6_ck400_l4_a448sr_never"),
        ("plan 448", "plan 448x256 + render 4 nodes", V0, "plhv2_plan6_ck400_l4a114688_st0_never"),
        ("plan+SR only", "plan + SR only (no render)", SR, "plansr_l4")]
for task, sn in (("sand_filling_up_jar", "jar"), ("sandcastle_wave_impact", "castle")):
    for sd in (1234, 4321, 9999):
        items = ["%s %.0fs=%s/%s/%s_s%d/video.mp4" % (lab, T[k]["total"], root, task, arm, sd) for lab, k, root, arm in ARMS]
        subprocess.run([S + "/venv/bin/python", S + "/scripts/concat_labeled.py", "%s/%s__seed%d__plan_render_speedup.mp4" % (P, sn, sd), "--height", "200"] + items, check=True, capture_output=True)
B = S + "/paper/overnight/blind_rfast"; mp = json.load(open(B + "/mapping.json")); V = collections.defaultdict(list)
for f in glob.glob(B + "/r*_p*.json"):
    for r in json.load(open(f)): V[r["id"]].append(r["drift"])
C = {(mp[h]["task"], mp[h]["arm"], mp[h]["seed"]): sum(v) * 2 < len(v) for h, v in V.items()}
ref = ARMS[0][3]; rows = []
for lab, k, root, arm in ARMS:
    j = sum(c for (t, a, s), c in C.items() if a == arm and "jar" in t); c_ = sum(c for (t, a, s), c in C.items() if a == arm and "castle" in t)
    g = sum(1 for (t, a, s), c in C.items() if a == arm and c and not C[(t, ref, s)]); l = sum(1 for (t, a, s), c in C.items() if a == arm and not c and C[(t, ref, s)])
    rows.append((lab, T[k], j, c_, g, l))
best = sorted({r[2] + r[3] for r in rows}, reverse=True)
fmt = lambda v, a: ("**%s**" % v) if a == best[0] else (("<u>%s</u>" % v) if len(best) > 1 and a == best[1] else str(v))
tt = sorted({round(r[1]["total"], 1) for r in rows}); ft = lambda v: ("**%.1f**" % v) if round(v, 1) == tt[0] else (("<u>%.1f</u>" % v) if round(v, 1) == tt[1] else "%.1f" % v)
md = ["| variant | plan (s) | render (s) | SR (s) | total ↓ (s) | clean jar ↑ /19 | clean castle ↑ /19 | clean total ↑ /38 | flips vs current (+ / −) |", "|---|---|---|---|---|---|---|---|---|"]
for lab, t, j, c_, g, l in rows:
    md.append("| %s | %.1f | %.1f | %.1f | %s | %d | %d | %s | %s |" % (lab, t["plan"], t["render"], t["sr"], ft(t["total"]), j, c_, fmt(j + c_, j + c_), "-" if lab.startswith("current") else "+%d / −%d" % (g, l)))
open(P + "/README.md", "w").write("""# plan -> render speed-ups (10/2)

Plan = LingBot v2 14B bidirectional, public 4-step LoRA (l4), no CFG, 576x304 unless noted. Render = LingBot v2 14B causal-fast + plan-hint LoRA, 832x480.
SR = Real-ESRGAN realesr-general-x4v3 per frame to 832x480. Times on one GPU, model loading excluded. jar + castle, 19 seeds each (1234, 4321, 9999, 101-116).
Quality = 3 blind raters, majority 'no drift' (strict rubric: new objects, teleports, colour/texture collapse, freeze, lighting change), same batch for all arms.
Videos: 3 seeds x 2 scenes, panels left to right in the table order.

TABLE
""".replace("TABLE", "\n".join(md)))
print("\n".join(md))
