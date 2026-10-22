"""Collect everything presentable for the 9/28 action-item review into paper/0928/ with readable names. Re-runnable (copies only missing/updated files)."""
import os, shutil, json, glob
S = "/home/hk28965/specwm"; V0 = S + "/lingbot-world-v2/specwm_out/spec_v0"; D = S + "/paper/0928"; SCR = "/tmp/claude-1741758895/-home-hk28965/2af3c383-3218-4bda-a425-8a247f5f536a/scratchpad"
def cp(src, dst):
    if not os.path.exists(src): return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst): shutil.copy2(src, dst)
    return True
TASK = {"sand_filling_up_jar": "jar", "sandcastle_wave_impact": "castle"}
ARM = {"base70": "base70_target", "never": "fast_alone", "plh_plan6_ck400_cr_st0_never": "v2safe", "plh_plan6_ck400_l4c_st0_never": "v2fastCFG", "plh_plan6_ck400_l4_st0_never": "v2fast",
       "specstep_n012c9_never": "exactspec_cfg5", "tgtonly_n012c9_never": "nullarm_cfg5", "specstep_n012c9g1_never": "exactspec_cfg1", "tgtonly_n012c9g1_never": "nullarm_cfg1", "specstep_n012c9g2p5_never": "exactspec_cfg2p5",
       "plkl_kv4_ck800_cr_never": "kv4_ck800_render", "plan_cr": "plan_cr_lowres", "plan_l4c": "plan_l4c_lowres",
       "sdvg_t-0p7_ir": "sdvg_port_tau-0.7", "sdvg_t-2p2_ir": "sdvg_port_tau-2.2", "sdvg_t-2p5_ir": "sdvg_port_tau-2.5"}
SEEDS = [1234, 4321, 9999, 11, 22, 33, 44, 55]
# 1. report + docs
cp(S + "/paper/action_items_report.md", D + "/00_report/action_items_report.md"); cp(SCR + "/report/action_items.html", D + "/00_report/action_items_report.html")
for f, dst in [("item2_email.md", "item2_speculative_baseline/email_to_authors_DRAFT.md"), ("item2_repro_plan.md", "item2_speculative_baseline/reproduction_plan.md"), ("sdvg_baseline.md", "item2_speculative_baseline/sdvg_baseline_branch_verdict.md"),
               ("item3_design.md", "item3_exact_ar_speculative/design_and_proof.md"), ("designs_with_verdicts.json", "item3_exact_ar_speculative/three_designs_with_adversarial_verdicts.json"),
               ("item4_protocol.md", "item4_eval_protocol/protocol_review.md"), ("prereg/PREREG_item4.md", "item4_eval_protocol/PREREG_confirmatory_batch.md"),
               ("item5_design_space.md", "item5_verifier_design_space/design_space.md"), ("verifier_space.json", "item5_verifier_design_space/verifier_space.json"),
               ("literature.json", "item2_speculative_baseline/literature_survey.json"), ("audit_summary.md", "item3_exact_ar_speculative/code_audit_summary.md"), ("open_questions.md", "00_report/open_questions.md")]:
    cp(S + "/paper/overnight/" + f, D + "/" + dst)
cp(S + "/lingbot-world-v2/specwm_out/onestep_grid.json", D + "/item6_onestep_cost/onestep_cost_grid.json")
cp(S + "/lingbot-world-v2/specwm_out/spec_v0/NIGHT3_NOTES.md", D + "/00_report/lab_notes_NIGHT3_NOTES.md")
# 2. blind protocol data (+ de-anonymised sheets)
B = S + "/paper/overnight/blind"; mp = json.load(open(B + "/mapping.json"))
for f in ["blind_scores.json", "blind_cats.json", "mapping.json", "results_batch1.json", "results_batch2.json", "results_batch3.json", "results_batch4.json"]: cp(B + "/" + f, D + "/item4_eval_protocol/blind_labels/" + f)
for hid, m in mp.items():
    if m["arm"] in ARM and m["task"] in TASK: cp("%s/%s.jpg" % (B, hid), "%s/item4_eval_protocol/contact_sheets/%s_s%d_%s.jpg" % (D, TASK[m["task"]], m["seed"], ARM[m["arm"]]))
# 3. videos + run json
n = 0
for t, tn in TASK.items():
    for arm, an in ARM.items():
        for sd in SEEDS:
            src = "%s/%s/%s_s%d" % (V0, t, arm, sd)
            if cp(src + "/video.mp4", "%s/videos/%s/%s_s%d_%s.mp4" % (D, an.split("_")[0] if an.startswith(("v2", "sdvg", "exactspec", "nullarm")) else an, tn, sd, an)): n += 1
            for j in ("spec_v0.json", "plan.json"): cp(src + "/" + j, "%s/videos/_run_json/%s_s%d_%s.json" % (D, tn, sd, an))
# 4. item 2/3 exact sampler evidence, item 5/6 extras, kv4
for f in glob.glob(S + "/exp/kveval/*"): cp(f, D + "/appendix_kv4_translator/" + os.path.basename(f))
cp(S + "/exp/kveval/ir_pool.json", D + "/item2_speculative_baseline/imagereward_calibration_pool.json")
for f in glob.glob(SCR + "/diagram/*.pptx") + glob.glob(SCR + "/diagram/*.html"): cp(f, D + "/00_report/diagram_" + os.path.basename(f))
# 5. README index
arts = "- Report page: https://claude.ai/artifact/5Ufm4yR4HAJViNa9KUByho\n- Latest results (videos side by side): https://claude.ai/artifact/LdmwkoAaRbTkgYWgYG8y7J\n- Five StEvo prompts: https://claude.ai/artifact/C5hm8PX8eEGMqZcCs8jHmV\n- Easy scenes: https://claude.ai/artifact/DXCqTSUQhzm3gWdo1BbjjK\n- Draft–verify diagram: https://claude.ai/artifact/AFDBv7ijNTcRM4gj9X2Wfu\n"
readme = """# 0928 — action-item results (collected %s)

%s
| folder | action item | contents |
|---|---|---|
| 00_report | all | action_items_report.md/.html (the morning report), lab notes (NIGHT3_NOTES.md), open questions, draft–verify diagram (pptx/html) |
| item2_speculative_baseline | 2 | email draft to De Bortoli et al., reproduction plan, SDVG baseline: branch verdict (`dev` = paper) + ImageReward calibration pool |
| item3_exact_ar_speculative | 3 | design + exactness proof sketch, 3 designs with adversarial verdicts, code audit |
| item4_eval_protocol | 4 | protocol review, pre-registration, blind labels (3 raters, 85 clips), de-anonymised contact sheets `<task>_s<seed>_<arm>.jpg` |
| item5_verifier_design_space | 5 | verifier/correction design space table and JSON |
| item6_onestep_cost | 6 | one-step Base cost grid (480p vs 576×304, 6/12/21 latents, ± CFG) |
| videos/<group> | 1, 3, 4 | `<task>_s<seed>_<arm>.mp4` for Base-70 / Fast alone / v2-safe / v2-fast-CFG / v2-fast / exact spec (CFG 5, 2.5, 1) / null arm / kv4 / low-res plans / SDVG port (τ sweep, as they finish); `_run_json` = per-run spec_v0.json (plan string, chunk log, acceptance, timings) |
| appendix_kv4_translator | (data expansion) | kv4/kv5 metric tables and frame strips vs v2-safe and Base |

Arms: base70_target = Base-70 (Wan2.2 I2V A14B, 70 steps, bidirectional); fast_alone = LingBot-World v2 14B causal (4 nodes); v2safe = 16-step Base low-res plan + Fast plan-hint render; v2fastCFG = 6-step LoRA plan (CFG 3 on 2 steps) + same render; v2fast = 4-step LoRA plan; exactspec_cfgX = exact reflection-coupling speculative sampler with the one-step Base CFG-X chunk target; nullarm_cfgX = the same target without a draft; kv4_ck800_render = KV-translator route; sdvg_port_tau = SDVG router (ImageReward worst-frame ≥ τ) with Fast draft and Base correction.
Videos copied: %d. Seeds 1234/4321/9999 for every arm, 11/22/33/44/55 additionally for the protocol arms.
""" % (__import__("time").strftime("%Y-%m-%d %H:%M CDT", __import__("time").localtime(__import__("time").time() - 5 * 3600)), arts, n)
open(D + "/README.md", "w").write(readme); print("videos", n)
