#!/usr/bin/env python3
"""StEvo-Bench 결과표: runs/<model>/summary.json (+ per_task 리포트) -> specwm_out/stevo/stevo_results.md"""
import json, glob, os
R = "/home/hk28965/specwm/lingbot-world-v2/specwm_out/stevo"; rows = []
NAMES = {"lingbot_fast": ("LingBot Fast 14B (v2 causal-fast)", "camera lookaway s-3,right-30,left-30 (253 f)"), "lingbot_base_flat": ("LingBot Base v1 (A14B, 70-step)", "camera lookaway (253 f)"),
         "lingbot_v2_1p3b": ("LingBot v2 1.3B causal-fast", "camera lookaway (253 f)"), "lingbot_v2_pretrain_360_173": ("LingBot v2 causal-pretrain 14B", "camera lookaway s-3,right-10,left-20,right-10 (173 f), 360x512"),
         "h3world_lingbotcam": ("H3-World (H3 + LoRA)", "camera lookaway as keyboard pan (243 f)"), "h3_video": ("MiniMax-H3", "video-model protocol: occlusion / light-off prompt (243 f)"),
         "lingbot_reground_fast_alone": ("Fast alone (regrounding control, W18/sink6, seed 1234)", "camera lookaway (161 f)"), "lingbot_reground_resync_k6": ("Fast + ideal regrounding (Base70 ctx 0-5 in KV)", "camera lookaway (161 f)"),
         "lingbot_reground_kvrw_laundered": ("Fast + ours: KV-rewrite from laundered+RePaint target", "camera lookaway (161 f)"), "longlive2_5b_video": ("LongLive-2.0-5B (no camera control)", "video-model protocol: occlusion / light-off prompt"),
         "lingbot_fast_360_2step": ("Fast 360p (304x576) 2-step (+WC k3, 0 skips)", "camera lookaway (249 f)"), "lingbot_fast_360_2step_wcforce": ("Fast 360p 2-step + WC forced skip", "camera lookaway (249 f)"), "lingbot_fast_480_2step": ("Fast 480p 2-step (axis: steps only)", "camera lookaway (249 f)"), "lingbot_fast_360_4step": ("Fast 360p 4-step (axis: resolution only)", "camera lookaway (249 f)"), "lingbot_fast_360_2step_lanczos": ("Fast 360p 2-step + Lanczos → 464x848", "camera lookaway (249 f)"), "lingbot_fast_360_2step_esrgan": ("Fast 360p 2-step + Real-ESRGAN compact → 464x848", "camera lookaway (249 f)"), "lingbot_fast_360_2step_latentup": ("Fast 360p 2-step + latent 1.5x upsample decode", "camera lookaway (249 f)"), "lingbot_anchor_target480": ("Fast + anchor restart (corrected-target frame 72)", "camera lookaway (253 f composite)"), "lingbot_anchor_fast": ("Fast + restart from own frame 72 (control)", "camera lookaway (253 f composite)")}
def rate(per, key, sub=None):
    vals = []
    for p in glob.glob(per + "/*/" + key + "_report*.json"):
        d = json.load(open(p)); v = d.get(sub) if sub else None
        if v is not None: vals.append(bool(v))
    return (100.0 * sum(vals) / len(vals), len(vals)) if vals else (None, 0)
for m in ["lingbot_fast", "lingbot_base_flat", "lingbot_v2_1p3b", "lingbot_v2_pretrain_360_173", "h3world_lingbotcam", "h3_video", "lingbot_reground_fast_alone", "lingbot_reground_resync_k6", "lingbot_reground_kvrw_laundered", "longlive2_5b_video", "lingbot_fast_360_2step", "lingbot_fast_480_2step", "lingbot_fast_360_4step", "lingbot_fast_360_2step_lanczos", "lingbot_fast_360_2step_esrgan", "lingbot_fast_360_2step_latentup", "lingbot_anchor_target480", "lingbot_anchor_fast"]:
    per = R + "/runs/%s/per_task" % m; name, proto = NAMES[m]
    if not os.path.isdir(per): rows.append((name, proto, "TBD", "TBD", "TBD", "TBD", "TBD", "TBD", "TBD")); continue
    occ, n = rate(per, "control", "occlusion_done"); trg, _ = rate(per, "control", "trigger_applied"); ctrl = None
    prog, n2 = rate(per, "se", "state_evol"); phys, n3 = rate(per, "physics", "physical_inaccuracy")
    tasks = {}
    for p in glob.glob(per + "/*/control_report*.json"):
        t = os.path.basename(os.path.dirname(p)); c = json.load(open(p)); tasks[t] = dict(ctrl=bool(c.get("occlusion_done")) and bool(c.get("trigger_applied")))
    for p in glob.glob(per + "/*/se_report*.json"): tasks.setdefault(os.path.basename(os.path.dirname(p)), {})["prog"] = json.load(open(p)).get("state_evol")
    for p in glob.glob(per + "/*/physics_report*.json"): tasks.setdefault(os.path.basename(os.path.dirname(p)), {})["phys_ok"] = not json.load(open(p)).get("physical_inaccuracy", True)
    succ = [t for t, v in tasks.items() if v.get("ctrl") and v.get("prog") and v.get("phys_ok")]
    f = lambda x: "TBD" if x is None else "%.1f" % x
    ctrl = 100.0 * sum(1 for v in tasks.values() if v.get("ctrl")) / max(len(tasks), 1) if tasks else None
    rows.append((name, proto, str(n), f(occ), f(trg), f(ctrl), f(prog), f(None if phys is None else 100 - phys), "%.1f" % (100.0 * len(succ) / max(len(tasks), 1)) if tasks else "TBD"))
out = ["# StEvo-Bench results (judge: Qwen3-VL-32B-Instruct local, 3-vote majority; paper: Gemini 3.1 Pro)", "",
       "| model | protocol | N | Observation ctrl % | Action ctrl % | Control success % | State progress % | Physics plausible % | Task success % |", "|---|---|---|---|---|---|---|---|---|"]
out += ["| " + " | ".join(r) + " |" for r in rows]
out += ["| *paper: LingBot (Gemini)* | camera lookaway | 225 | 35.6 | 67.8 | – | 3.4 | 40.7 | 0.0 |", "| *paper: Veo 3 (Gemini)* | occlusion prompt | 225 | 81.6 | 88.0 | – | 17.4 | 82.6 | 8.7 |",
        "", "Notes: released eval code merges the paper's physics+coherence judges into one physics judge (no separate coherence column); task success = control ∧ progress ∧ physics-plausible."]
open(R + "/stevo_results.md", "w").write("\n".join(out)); print("\n".join(out))
