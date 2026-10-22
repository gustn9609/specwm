#!/usr/bin/env python3
"""Build the 'Correction Lineup' comparison page: every current rollout of the two test tasks next to Base 70-step and Fast alone,
with the measured generation time. Videos are transcoded small (h264, 448 px wide) into <out>/v/.
usage: build_compare_site.py OUT_DIR"""
import sys, os, json, subprocess
V0 = "/home/hk28965/specwm/lingbot-world-v2/specwm_out/spec_v0"
OUT = sys.argv[1]; os.makedirs(OUT + "/v", exist_ok=True)
TASKS = [("sandcastle_wave_impact", "castle", "Wave hits a sandcastle"), ("sand_filling_up_jar", "jar", "Sand filling a jar")]
SEEDS = [1234, 4321, 9999]
# (key, run dir prefix, label, group, short description)
METHODS = [
    ("base", "base70", "Base 70-step", "ref", "Target model, 70 steps, CFG 5"),
    ("fast", "never", "Fast alone", "ref", "Draft model only, 4 steps"),
    ("s16", "viz_best", "Ours · 16-step correction", "ours", "Semantic gate + Base 16-step RePaint"),
    ("cr", "cfg6reuse_semc100", "Ours · CFG 6 + flow reuse", "ours", "CFG on the first 6 steps, flow reused at steps 9/11/13"),
    ("ml", "mixlora46_cr_semc100", "Ours · + mixed prefix + LoRA", "ours", "Old prefix as 4x merged tokens read by the mixed-memory LoRA"),
    ("pl2", "plh_plan3_ck325_cr_st0_never", "Ours · plan → render v2", "ours", "One GPU, no Base call per chunk: Base plans the whole clip at 576×304 (16 steps, CFG on 6, 3 flows reused, ~133 s); Fast renders 480p with a plan-hint LoRA on its first two nodes only, trained on hints built exactly like the test-time plan (~73 s). Time = plan + render."),
    ("pl", "pl_plan300_never", "Ours · plan → render v1", "ours", "First version: Base plans at 576×304 (16 steps, ~300 s), Fast + plan-hint LoRA renders 480p (~75 s); the hint also reached the un-adapted nodes (colour drift). Time = plan + render."),
    ("plancr", "plan_cr", "Base low-res plan (v2)", "ref", "Base 16-step whole-clip generation at 576×304 with CFG on the first 6 steps and 3 reused flows — the plan v2 follows"),
    ("plan", "base16_a184320", "Base low-res plan (v1)", "ref", "Base 16-step whole-clip generation at 576×304 — the plan v1 follows"),
    ("bg", "bg1_n01", "Ours · Base-guided Fast", "ours", "Base's x0 blended into Fast's first two nodes (w=1), no corrections, no cuts"),
    ("fd", "fd1_cr_semc100", "Ours · from-draft σ 0.94", "other", "Base starts from Fast's node-1 state (σ 0.94) and runs only its low-noise steps"),
    ("mx", "mixed46_semc100", "Mixed prefix, no LoRA", "other", "Old prefix 4x merged, no adapter"),
    ("l4", "lora1022_semc100", "4-step distill LoRA", "other", "Public Wan2.2 4-step LoRA on Base, no CFG"),
    ("h2", "hand2_semc100", "Handoff to Fast", "other", "Base high-expert steps only, Fast finishes"),
    ("ag", "ag2_cr_semc100", "Autoguidance w=2", "other", "Fast velocity replaces the CFG uncond branch"),
    ("ac", "act90_cr_semc100", "Action-aware schedule 90°", "other", "Forced correction every 90° of camera yaw"),
]
NOTES = {
    ("castle", 1234): dict(base="Castle washed away; flat wet sand, foam", fast="Blue sky, forested headland", s16="Mountains and a turquoise sea; trees at the end", fd="No cuts, but the camera ends up under water", cr="Car and pier appear at chunk 9 (accepted Fast drafts)", ml="No car; foamy surf, low hills, ends on flat water", mx="Same as 16-step until chunk 8; clearer sky", l4="Cloudy, flat sand; dune-like terrain mid-way", ac="No car, but the castle keeps re-appearing"),
    ("castle", 4321): dict(base="Castle survives mid-video, washed away at the end", pl="Castle washed away, beach and hills — follows the plan", plan="Castle washed away, beach, hills", fast="Castle gone at once; tree line, bright blue sky, turquoise surf", s16="Castle gone at once; forested headland", cr="Headland and pine forest, greyer sky", ml="Open beach, pine islands at the edges; no hallucination", fd="No hallucinations and no cuts, but mostly foam close-ups; castle and beach are not recovered", mx="Castle survives mid-video (partly a different random path)", l4="Dark creatures running on the beach", h2="Green bushes and mountains later", ag="Pink haze and sand spray from chunk 1"),
    ("castle", 9999): dict(base="Castle stands; a wave arrives at the end", fast="Bright blue sky, open beach, forest line", s16="Castle fragments; mountains and forest", fd="A person walks along the beach", cr="Castle persists like Base; dark cavity at the end", ml="Castle disappears after chunk 5; flat beach, small dark object", mx="Castle whole; big wave at the end", l4="Dunes, then a concrete road-like strip"),
    ("jar", 1234): dict(base="Camera turns away to the table and wall", fast="Leaves the table: wooden room, then a man in a suit", s16="Coffee-coloured sand in chunks 9-12", bg="One jar, stable background, sand rises; does not turn away like Base", fd="Jar tips over and slides; flowers appear", cr="Single jar, consistent sand", ml="Jar looks empty with clear water in chunks 9-11", mx="Yellow liquid and foam in the last chunk", l4="Camera turns like Base; jar shrinks and blurs"),
    ("jar", 4321): dict(base="Static jar filling with sand", fast="Leaves the table: a blue room with windows", s16="A stick and coffee-coloured sand", fd="Leaves the table: concrete wall and doors", cr="Static jar filling — very Base-like", ml="Static jar until the last chunk, which turns into green/yellow liquid bottles", mx="Scene change: plates, bowl of yellow liquid", l4="Flying objects, a chair"),
    ("jar", 9999): dict(base="A teal pot appears, then an empty table", pl="Turns away to table and wall like Base; wood grain swims", plan="Turns away; no jar re-appearance", fast="Leaves the table: outdoors, a white house and a street", s16="Coloured straws appear", fd="Bottle, then a deer and a hedge", cr="Single jar, sand kept", ml="Red liquid in chunk 6 only; sand again from chunk 9", mx="Red liquid from chunk 6 to the end", l4="Jar duplicated and blurred at the edges", h2="Glowing liquid; contents change"),
}
def gen_s(d):
    for f in ("spec_v0.json", "revisit.json", "plan.json"):
        p = os.path.join(d, f)
        if os.path.exists(p): return round(json.load(open(p))["gen_s"])
    return None
def transcode(src, dst):
    if os.path.exists(dst) and os.path.getmtime(dst) > os.path.getmtime(src): return
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", "scale=448:-2", "-r", "16", "-c:v", "libx264", "-preset", "slow", "-crf", "31",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst], check=True)
cases = []
for task, short, name in TASKS:
    for sd in SEEDS:
        runs = []
        for key, pre, label, grp, desc in METHODS:
            d = os.path.join(V0, task, "%s_s%d" % (pre, sd)); src = os.path.join(d, "video.mp4"); t = gen_s(d); est = False
            if key == "fast" and not os.path.exists(src):   # castle 9999: the probe2 run has the Fast-alone video (its time includes Base probes)
                d2 = os.path.join(V0, task, "probe2_s%d" % sd); src = os.path.join(d2, "video.mp4"); t = 62; est = True
            smoke = False
            if key == "bg" and not os.path.exists(src):   # full run pending: show the 61-frame smoke test where one exists
                d2 = os.path.join(V0, task, "_smoke_bg1_never61_s%d" % sd)
                if os.path.exists(os.path.join(d2, "video.mp4")): d = d2; src = os.path.join(d2, "video.mp4"); t = gen_s(d2); smoke = True
            if not os.path.exists(src):
                if key in ("ml", "fd", "bg", "pl", "pl2", "plan", "plancr"): runs.append(dict(key=key, label=label, group=grp, desc=desc, pending=True))
                continue
            fn = "v/%s_%d_%s.mp4" % (short, sd, key); transcode(src, os.path.join(OUT, fn))
            plan = None
            pj = os.path.join(d, "spec_v0.json")
            if os.path.exists(pj) and key not in ("base", "fast", "plan"): plan = json.load(open(pj)).get("plan")
            note = NOTES.get((short, sd), {}).get(key, "")
            PLANDIR = {"pl": "base16_a184320", "pl2": "plan_cr"}
            if key in PLANDIR and t is not None:
                tp = gen_s(os.path.join(V0, task, "%s_s%d" % (PLANDIR[key], sd)))
                if tp is not None: note = "plan %d s + render %d s. " % (tp, t) + note; t = t + tp
            if smoke: note = "61-frame smoke test (4 chunks), time is for 61 frames — full 161-frame run pending. " + note
            runs.append(dict(key=key, label=label, group=grp, desc=desc, video=fn, sec=t, est=est, plan=plan, note=note, smoke=smoke))
        cases.append(dict(id="%s_%d" % (short, sd), short=short, name=name, seed=sd, runs=runs))
        print(short, sd, [r["key"] + ("*" if r.get("pending") else "") for r in runs], flush=True)
DATA = dict(cases=cases, methods=[dict(key=k, label=l, group=g) for k, _, l, g, _ in METHODS])
tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_site_template.html")).read()
open(os.path.join(OUT, "correction_lineup.html"), "w").write(tpl.replace("/*__DATA__*/null", json.dumps(DATA)))
tot = sum(os.path.getsize(os.path.join(OUT, "v", f)) for f in os.listdir(OUT + "/v"))
print("videos %d, %.1f MB" % (len(os.listdir(OUT + "/v")), tot / 1e6))
