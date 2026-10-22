# 0928 — action-item results (collected 2026-09-30 01:59 CDT)

- Report page: https://claude.ai/artifact/5Ufm4yR4HAJViNa9KUByho
- Latest results (videos side by side): https://claude.ai/artifact/LdmwkoAaRbTkgYWgYG8y7J
- Five StEvo prompts: https://claude.ai/artifact/C5hm8PX8eEGMqZcCs8jHmV
- Easy scenes: https://claude.ai/artifact/DXCqTSUQhzm3gWdo1BbjjK
- Draft–verify diagram: https://claude.ai/artifact/AFDBv7ijNTcRM4gj9X2Wfu

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
Videos copied: 141. Seeds 1234/4321/9999 for every arm, 11/22/33/44/55 additionally for the protocol arms.
