# All-v2 pipeline (draft = LingBot v2 14B causal-fast; planner / verifier / reference = LingBot v2 14B bid), jar & castle, seeds 1234/4321/9999

File names `<scene>__seed<seed>__<arm>__<seconds>s.mp4`. Reference = v2 bid, 70 steps (2,757-2,850 s). v1 Base-70 and the v1-plan render are included for comparison.

Metrics vs the v2 reference (6 cases): DINO distance v1 Base-70 0.399 | Fast 0.710 | v1 plan+render 0.668 | v2 plan+render 0.394 | + p/q verify + redraw 0.395. Loop closure at frame ~133 (reference 0.28): v1 plan 0.90, v2 plan 0.31.
Blind (3 raters): v2 reference severe-clean 5/6, v2 plan+render 3/6, + verify/redraw 3/6.

## Added 9/30 10:15 AM CDT: l4 planner (4-step LoRA 1.0, NO CFG) — the new v2 default
`2b_v2bid_plan_l4_noCFG_lowres` = plan (42 s), `3b_v2plan_l4_render` = Fast + hint LoRA render on it (73 s). Compare with `2_`/`3_` (l4c: 6-step LoRA 0.8 + CFG 2, the old default).
Blind, severe-clean, same batches: 22 base cases l4 18–19/22 vs l4c 15–17/22; held-out seeds 109–116 l4 11/16 vs l4c 8/16; pooled 38 cases l4 30/38 (castle 19/19, jar 11/19) vs l4c 25/38. Castle colour collapse disappears without CFG; jar teleports (~40 %) remain.
