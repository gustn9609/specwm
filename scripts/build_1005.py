"""Assemble paper/1005 from the clean re-runs: comparison videos (one per scene/seed) + README time tables.
 step1_sdvg_selfforcing   : drafter | SDVG tau -0.7 | SDVG tau 1.3 | target          (SDVG original code, Self-Forcing 1.3B -> Krea 14B, T2V, ImageReward)
 step2_lingbot_imagereward: drafter | SDVG-IR tau -0.7 | SDVG-IR tau -2.2 | target    (LingBot v2 1.3B -> 14B, action-conditioned, decode + ImageReward)
 step3_lingbot_latent     : drafter | latent critic tau 0.05 | tau -3.87 | target     (same loop, drift critic on latents, no decode)
 sdvg_steps_compare       : step1 SDVG tau -0.7 | step2 SDVG-IR tau -0.7 | step3 latent critic tau 0.05
 step3_lowres             : latent critic at 832x480 | 576x304 | 448x256 | 384x224 (raw, panels resized for display)
 step3_lowres_sr          : 832x480 | 576x304+SR | 448x256+SR | 384x224+SR (Real-ESRGAN x4v3 -> 832x480)
 sdvg_original_5methods   : 4 prompts x 3 seeds, drafter | tau -0.7 | 1.3 | 1.5 | target (pure drafter/target arms)
usage: python scripts/build_1005.py"""
import os, json, glob, subprocess, numpy as np
S = "/home/hk28965/specwm"; O = S + "/lingbot-world-v2/specwm_out"; P = S + "/paper/1005"; PY = S + "/venv/bin/python"
SR0 = O + "/sdvg_orig1005"; L = O + "/loop1005"; CP = "critic_noprev_noplanrc"
SC = [("sand_filling_up_jar", "jar"), ("sandcastle_wave_impact", "castle")]; SEEDS = [1234, 4321, 9999]
def sdvg(task, tag, sd):
    d = "%s/%s/%s_s%d" % (SR0, task, tag, sd); j = json.load(open(d + "/sdvg.json")) if os.path.exists(d + "/sdvg.json") else {}
    v = (glob.glob(d + "/*.mp4") or [None])[0]; return v, j.get("gen_s"), j.get("accepted"), j.get("rejected")
def loop(root, task, arm, sd):
    d = "%s/%s/%s_s%d" % (root, task, arm, sd); j = json.load(open(d + "/loop.json")) if os.path.exists(d + "/loop.json") else {}
    sr = json.load(open(d.replace(root, root + "_sr") + "/sr_timing.json")) if os.path.exists(d.replace(root, root + "_sr") + "/sr_timing.json") else {}
    return dict(video=d + "/video.mp4", sr_video=d.replace(root, root + "_sr") + "/video.mp4", gen=j.get("gen_s"), dec=j.get("decode_s"), acc=j.get("accepted"), n=j.get("chunks"), times=j.get("times", {}), sr=sr.get("sr_s"))
def concat(out, items):
    items = [(l, v) for l, v in items if v and os.path.exists(v)]
    if len(items) < 2: return
    subprocess.run([PY, S + "/scripts/concat_labeled.py", out] + ["%s=%s" % (l, v) for l, v in items], check=True, capture_output=True)
f = lambda x: "-" if x is None else "%.1f" % x
README = {}
# ---- step 1
md = ["| scene | seed | arm | time ↓ (s, incl. decode) | blocks accepted / rejected |", "|---|---|---|---|---|"]; os.makedirs(P + "/step1_sdvg_selfforcing", exist_ok=True)
for task, sn in SC:
    for sd in SEEDS:
        arms = [("drafter only", "pure_draft"), ("SDVG tau -0.7", "sdvg_t-0p7"), ("SDVG tau 1.3", "sdvg_t1p3"), ("target only", "pure_target")]; items = []
        for lab, tag in arms:
            v, g, a_, r_ = sdvg(task, tag, sd); items.append(("%s  %ss" % (lab, f(g)), v)); md.append("| %s | %d | %s | %s | %s / %s |" % (sn, sd, lab, f(g), a_, r_))
        concat("%s/step1_sdvg_selfforcing/%s__seed%d__drafter-SDVG0.7-SDVG1.3-target.mp4" % (P, sn, sd), items)
README["step1_sdvg_selfforcing"] = md
# ---- steps 2, 3
for sub, arms in (("step2_lingbot_imagereward", [("drafter only", "none"), ("SDVG-IR tau -0.7", "ir_-0.7"), ("SDVG-IR tau -2.2", "ir_-2.2"), ("target only", "target")]),
                  ("step3_lingbot_latent", [("drafter only", "none"), ("latent critic tau 0.05", CP + "_0.05"), ("latent critic tau -3.87", CP + "_-3.87"), ("target only", "target")])):
    os.makedirs(P + "/" + sub, exist_ok=True); md = ["| scene | seed | arm | latent gen ↓ (s) | + final decode (s) | total ↓ (s) | chunks accepted | draft / verify / target / KV commit (s) |", "|---|---|---|---|---|---|---|---|"]
    for task, sn in SC:
        for sd in SEEDS:
            items = []
            for lab, arm in arms:
                r = loop(L, task, arm, sd); tot = None if r["gen"] is None else r["gen"] + (r["dec"] or 0); items.append(("%s  %ss" % (lab, f(tot)), r["video"]))
                t = r["times"]; md.append("| %s | %d | %s | %s | %s | %s | %s/%s | %s / %s / %s / %s |" % (sn, sd, lab, f(r["gen"]), f(r["dec"]), f(tot), r["acc"], r["n"], f(t.get("draft")), f(t.get("verify")), f(t.get("target")), f(t.get("commit"))))
            concat("%s/%s/%s__seed%d__%s.mp4" % (P, sub, sn, sd, "-".join(a for _, a in arms).replace(CP + "_", "critic")), items)
    README[sub] = md
# ---- SDVG step 1 vs 2 vs 3
os.makedirs(P + "/sdvg_steps_compare", exist_ok=True); md = ["| scene | seed | variant | total time ↓ (s) | accepted |", "|---|---|---|---|---|"]
for task, sn in SC:
    for sd in SEEDS:
        v1, g1, a1, _ = sdvg(task, "sdvg_t-0p7", sd); r2 = loop(L, task, "ir_-0.7", sd); r3 = loop(L, task, CP + "_0.05", sd)
        t2 = None if r2["gen"] is None else r2["gen"] + (r2["dec"] or 0); t3 = None if r3["gen"] is None else r3["gen"] + (r3["dec"] or 0)
        concat("%s/sdvg_steps_compare/%s__seed%d__step1-SelfForcing-IR__step2-LingBot-IR__step3-LingBot-latent.mp4" % (P, sn, sd),
               [("1 Self-Forcing+IR  %ss" % f(g1), v1), ("2 LingBot+IR  %ss" % f(t2), r2["video"]), ("3 LingBot+latent  %ss" % f(t3), r3["video"])])
        md += ["| %s | %d | 1: Self-Forcing 1.3B -> Krea 14B, ImageReward tau -0.7 (T2V) | %s | %s/13 |" % (sn, sd, f(g1), a1), "| %s | %d | 2: LingBot 1.3B -> 14B, ImageReward tau -0.7 (action) | %s | %s/%s |" % (sn, sd, f(t2), r2["acc"], r2["n"]),
               "| %s | %d | 3: LingBot 1.3B -> 14B, latent drift critic tau 0.05 (action) | %s | %s/%s |" % (sn, sd, f(t3), r3["acc"], r3["n"])]
README["sdvg_steps_compare"] = md
# ---- low resolution (raw and SR)
for sub, use_sr in (("step3_lowres", False), ("step3_lowres_sr", True)):
    os.makedirs(P + "/" + sub, exist_ok=True); md = ["| scene | seed | resolution | latent gen (s) | decode (s) | SR (s) | total ↓ (s) | chunks accepted |", "|---|---|---|---|---|---|---|---|"]
    for task, sn in SC:
        for sd in SEEDS:
            items = []
            for lab, root in (("832x480", L), ("576x304", O + "/loop1005_lr576"), ("448x256", O + "/loop1005_lr448"), ("384x224", O + "/loop1005_lr384")):
                r = loop(root, task, CP + "_0.05", sd); sr = r["sr"] if (use_sr and root != L) else None
                tot = None if r["gen"] is None else r["gen"] + (r["dec"] or 0) + (sr or 0)
                vid = r["sr_video"] if (use_sr and root != L) else r["video"]; items.append(("%s%s  %ss" % (lab, "+SR" if sr is not None else "", f(tot)), vid))
                md.append("| %s | %d | %s | %s | %s | %s | %s | %s/%s |" % (sn, sd, lab, f(r["gen"]), f(r["dec"]), f(sr), f(tot), r["acc"], r["n"]))
            concat("%s/%s/%s__seed%d__latent-critic_480-576-448-384%s.mp4" % (P, sub, sn, sd, "_SR" if use_sr else ""), items)
    README[sub] = md
# ---- SDVG original, 4 prompts x 5 methods
os.makedirs(P + "/sdvg_original_5methods", exist_ok=True); md = ["| prompt | seed | drafter only | SDVG tau -0.7 | SDVG tau 1.3 | SDVG tau 1.5 | target only |", "|---|---|---|---|---|---|---|"]
for task, sn in (("tokyo_street", "tokyo_street"), ("bigsur_drone", "bigsur_drone")) + tuple(SC):
    for sd in SEEDS:
        row = []; items = []
        for lab, tag in (("drafter only", "pure_draft"), ("SDVG -0.7", "sdvg_t-0p7"), ("SDVG 1.3", "sdvg_t1p3"), ("SDVG 1.5", "sdvg_t1p5"), ("target only", "pure_target")):
            v, g, a_, r_ = sdvg(task, tag, sd); items.append(("%s  %ss" % (lab, f(g)), v)); row.append("%s s (%s acc)" % (f(g), a_))
        concat("%s/sdvg_original_5methods/%s__seed%d__5methods.mp4" % (P, sn, sd), items); md.append("| %s | %d | %s |" % (sn, sd, " | ".join(row)))
README["sdvg_original_5methods"] = md
for sub, md in README.items():
    open("%s/%s/README.md" % (P, sub), "w").write("# %s\n\nTimes on dedicated GPUs (10/2 clean re-run), model loading excluded; 832x480 unless noted, 16 fps.\n\n%s\n" % (sub, "\n".join(md)))
print("[BUILD_1005_DONE]", {k: len(v) - 2 for k, v in README.items()})
