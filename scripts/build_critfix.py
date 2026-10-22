"""paper/1005/step3_lingbot_latent_fixed (+ step3_lowres_fixed when loop1005fix_lr* exist): latent drift critic with the fixed accept direction
(logit <= tau). Panels: drafter | critic tau 0.05 | critic tau -3.87 | target (832x480), and 832x480 | 576x304+SR | 448x256+SR | 384x224+SR.
Also marks the old step3 folders as produced by the inverted-accept bug. usage: python scripts/build_critfix.py"""
import os, json, subprocess
S = "/home/hk28965/specwm"; O = S + "/lingbot-world-v2/specwm_out"; P = S + "/paper/1005"; PY = S + "/venv/bin/python"; CP = "critic_noprev_noplanrc"
SC = [("sand_filling_up_jar", "jar"), ("sandcastle_wave_impact", "castle")]; SEEDS = [1234, 4321, 9999]
f = lambda x: "-" if x is None else "%.1f" % x
def loop(root, task, arm, sd, sr=False):
    d = "%s/%s/%s_s%d" % (root, task, arm, sd); j = json.load(open(d + "/loop.json")) if os.path.exists(d + "/loop.json") else {}
    ds = d.replace(root, root + "_sr"); st = json.load(open(ds + "/sr_timing.json")) if sr and os.path.exists(ds + "/sr_timing.json") else {}
    return dict(video=(ds if sr else d) + "/video.mp4", gen=j.get("gen_s"), dec=j.get("decode_s"), acc=j.get("accepted"), n=j.get("chunks"), plan=j.get("plan"), sr=st.get("sr_s"))
def concat(out, items):
    items = [(l, v) for l, v in items if v and os.path.exists(v)]
    if len(items) >= 2: subprocess.run([PY, S + "/scripts/concat_labeled.py", out] + ["%s=%s" % (l, v) for l, v in items], check=True, capture_output=True)
NOTE = "**Fixed 10/5:** the drift critic (higher = more drift) is accepted on logit <= tau. The old step3 folders accepted logit >= tau (inverted)."
sub = "step3_lingbot_latent_fixed"; os.makedirs("%s/%s" % (P, sub), exist_ok=True)
md = ["| scene | seed | arm | latent gen ↓ (s) | + decode (s) | total ↓ (s) | accepted | plan |", "|---|---|---|---|---|---|---|---|"]
for task, sn in SC:
    for sd in SEEDS:
        items = []
        for lab, root, arm in (("drafter only", O + "/loop1005", "none"), ("latent critic tau 0.05", O + "/loop1005fix", CP + "_0.05"), ("latent critic tau -3.87", O + "/loop1005fix", CP + "_-3.87"), ("target only", O + "/loop1005", "target")):
            r = loop(root, task, arm, sd); tot = None if r["gen"] is None else r["gen"] + (r["dec"] or 0); items.append(("%s  %ss" % (lab, f(tot)), r["video"]))
            md.append("| %s | %d | %s | %s | %s | %s | %s/%s | %s |" % (sn, sd, lab, f(r["gen"]), f(r["dec"]), f(tot), r["acc"], r["n"], r["plan"]))
        concat("%s/%s/%s__seed%d__drafter-critic0.05-critic-3.87-target.mp4" % (P, sub, sn, sd), items)
open("%s/%s/README.md" % (P, sub), "w").write("# %s\n\n%s\n\nPlan: T = chunk made by the 14B, D = draft chunk accepted (chunk 0 always T).\n\n%s\n" % (sub, NOTE, "\n".join(md)))
if os.path.exists(O + "/loop1005fix_lr384_sr"):
    sub = "step3_lowres_fixed_sr"; os.makedirs("%s/%s" % (P, sub), exist_ok=True); md = ["| scene | seed | resolution | latent gen (s) | decode (s) | SR (s) | total ↓ (s) | accepted |", "|---|---|---|---|---|---|---|---|"]
    for task, sn in SC:
        for sd in SEEDS:
            items = []
            for lab, root in (("832x480", O + "/loop1005fix"), ("576x304", O + "/loop1005fix_lr576"), ("448x256", O + "/loop1005fix_lr448"), ("384x224", O + "/loop1005fix_lr384")):
                use = root != O + "/loop1005fix"; r = loop(root, task, CP + "_0.05", sd, sr=use); tot = None if r["gen"] is None else r["gen"] + (r["dec"] or 0) + (r["sr"] or 0)
                items.append(("%s%s  %ss" % (lab, "+SR" if use else "", f(tot)), r["video"])); md.append("| %s | %d | %s | %s | %s | %s | %s | %s/%s |" % (sn, sd, lab, f(r["gen"]), f(r["dec"]), f(r["sr"]), f(tot), r["acc"], r["n"]))
            concat("%s/%s/%s__seed%d__latent-critic-fixed_480-576-448-384_SR.mp4" % (P, sub, sn, sd), items)
    open("%s/%s/README.md" % (P, sub), "w").write("# %s\n\n%s\n\n%s\n" % (sub, NOTE, "\n".join(md)))
for old in ("step3_lingbot_latent", "step3_lowres", "step3_lowres_sr", "sdvg_steps_compare"):
    p = "%s/%s/README.md" % (P, old); t = open(p).read()
    if "INVALID" not in t: open(p, "w").write("> **INVALID (10/5):** latent-critic arms here accepted the drift critic on logit >= tau (inverted; the most drifted chunks were accepted). See step3_*_fixed.\n\n" + t)
print("[BUILD_CRITFIX_DONE]")
