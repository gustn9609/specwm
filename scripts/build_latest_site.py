"""Latest-results page: for each test case, Base 70-step next to the newest plans and renders. usage: build_latest_site.py OUT"""
import os, sys, json, subprocess, html
S = "/home/hk28965/specwm"; V0 = S + "/lingbot-world-v2/specwm_out/spec_v0"; OUT = sys.argv[1]; os.makedirs(OUT + "/v", exist_ok=True)
SPEC = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
TASKS = [tuple(t) for t in SPEC.get("tasks", [("sandcastle_wave_impact", "castle", "Wave hits a sandcastle"), ("sand_filling_up_jar", "jar", "Sand filling a jar")])]; SEEDS = SPEC.get("seeds", [1234, 4321, 9999])
# (key, run prefix, label, group, description); plan time comes from the plan dir's json, render time from spec_v0.json; total = plan + render for renders
M = [
 ("base", "base70", "Base 70-step (target)", "ref", "Wan2.2 I2V A14B, 70 steps, CFG 5 — the look we want", None),
 ("plan_cr", "plan_cr", "Plan · 16-step (safe)", "plan", "Base at 576×304, 16 steps, CFG on 6, 3 flows reused", None),
 ("v2", "plh_plan6_ck400_cr_st0_never", "v2-safe render", "ours", "Fast + hint LoRA (plan6_ck400) on the 16-step plan; hint only at the adapted nodes", "plan_cr"),
 ("plan_l4", "plan_l4", "Plan · 4-step LoRA (fast)", "plan", "Base + public lightx2v 4-step LoRA at 576×304, no CFG", None),
 ("v2f", "plh_plan6_ck400_l4_st0_never", "v2-fast render", "ours", "Same render on the 4-step plan", "plan_l4"),
 ("kvy", "plky_kv3_ck400_cr_never", "y-hint + KV memory render", "ours", "kv3_ck400: hint LoRA plus the Base→Fast KV translator (plan memory at the adapted nodes)", "plan_cr"),
 ("kv", "plkl_kv1_ck1600_cr_never", "KV memory only render", "other", "kv1_ck1600: plan memory through the translator, no y hint", "plan_cr"),
 ("tex", "plt_plan6_ck400_tex3_ck248_cr_st0_never", "+ texture LoRA (failed)", "other", "tex3: on-policy texture LoRA on the low-noise nodes — sharp but frozen", "plan_cr"),
 ("plan_crw", "plan_crw", "Plan · 16-step, 2 windows (streaming)", "plan", "Two 81-frame windows, the second continued from the first's last frame", None),
 ("v2w", "plh_plan6_ck400_crw_st0_never", "v2 render on the streaming plan", "ours", "Render on the windowed plan", "plan_crw"),
 ("plan_l4c", "plan_l4c", "Plan · 6-step LoRA 0.8 + CFG 2", "plan", "Hallucination-mitigation variant of the fast planner", None),
 ("v2c", "plh_plan6_ck400_l4c_st0_never", "v2 render on the 6-step plan", "ours", "", "plan_l4c"),
 ("sdvg07", "sdvg_t-0p7_ir", "SDVG baseline τ = −0.7", "baseline", "Hu & Zhang 2026 (arXiv 2604.17397) router: Fast draft, ImageReward worst-frame ≥ τ accepts, first chunk forced to Base, rejected chunks Base 16-step", None),
 ("sdvg22", "sdvg_t-2p2_ir", "SDVG baseline τ = −2.2", "baseline", "Same router, τ near the ImageReward floor (accepts ~1/3 of Fast chunks)", None),
 ("sdvg25", "sdvg_t-2p5_ir", "SDVG baseline τ = −2.5", "baseline", "Same router, τ below the floor: only the forced first chunk is Base (force-reject-first ablation)", None),
]
if "methods" in SPEC: M = [tuple(m) for m in SPEC["methods"]]
def secs(d):
    for f in ("spec_v0.json", "revisit.json", "plan.json"):
        p = os.path.join(d, f)
        if os.path.exists(p): return json.load(open(p)).get("gen_s")
    return None
def transcode(src, dst):
    if os.path.exists(dst) and os.path.getmtime(dst) > os.path.getmtime(src): return
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", "scale=448:-2", "-r", "16", "-c:v", "libx264", "-preset", "slow", "-crf", "31", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst], check=True)
cases = []
for task, short, name in TASKS:
    for sd in SEEDS:
        runs = []
        for key, pre, label, grp, desc, plan_of in M:
            d = os.path.join(V0, task, "%s_s%d" % (pre, sd)); src = os.path.join(d, "video.mp4")
            if not os.path.exists(src): runs.append(dict(key=key, label=label, group=grp, desc=desc, pending=True)); continue
            fn = "v/%s_%d_%s.mp4" % (short, sd, key); transcode(src, os.path.join(OUT, fn)); t = secs(d); tp = None
            if plan_of:
                pd = os.path.join(V0, task, "%s_s%d" % (dict((m[0], m[1]) for m in M)[plan_of], sd)); tp = secs(pd)
            runs.append(dict(key=key, label=label, group=grp, desc=desc, video=fn, sec=t, plan_sec=tp))
        cases.append(dict(id="%s_%d" % (short, sd), name=name, seed=sd, runs=runs))
        print(short, sd, [r["key"] + ("*" if r.get("pending") else "") for r in runs], flush=True)
DATA = dict(cases=cases)
tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "latest_site_template.html")).read()
if "title" in SPEC: tpl = tpl.replace("<title>Plan → Render Latest</title>", "<title>%s</title>" % html.escape(SPEC["title"])).replace("<h1>Plan → Render Latest</h1>", "<h1>%s</h1>" % html.escape(SPEC["title"]))
if "sub" in SPEC: tpl = tpl.replace(tpl[tpl.index('<p class="sub">'):tpl.index('</p>', tpl.index('<p class="sub">')) + 4], '<p class="sub">%s</p>' % html.escape(SPEC["sub"]))
open(os.path.join(OUT, SPEC.get("out", "latest_results.html")), "w").write(tpl.replace("/*__DATA__*/null", json.dumps(DATA)).replace('localStorage.getItem("latest_case")', 'localStorage.getItem("%s")' % SPEC.get("out", "latest_case")).replace('localStorage.setItem("latest_case"', 'localStorage.setItem("%s"' % SPEC.get("out", "latest_case")))
tot = sum(os.path.getsize(os.path.join(OUT, "v", f)) for f in os.listdir(OUT + "/v")); print("videos %d, %.1f MB" % (len(os.listdir(OUT + "/v")), tot / 1e6))
