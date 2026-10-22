# plan -> render speed-ups (10/2)

Plan = LingBot v2 14B bidirectional, public 4-step LoRA (l4), no CFG, 576x304 unless noted. Render = LingBot v2 14B causal-fast + plan-hint LoRA, 832x480.
SR = Real-ESRGAN realesr-general-x4v3 per frame to 832x480. Times on one GPU, model loading excluded. jar + castle, 19 seeds each (1234, 4321, 9999, 101-116).
Quality = 3 blind raters, majority 'no drift' (strict rubric: new objects, teleports, colour/texture collapse, freeze, lighting change), same batch for all arms.
Videos: 3 seeds x 2 scenes, panels left to right in the table order.

| variant | plan (s) | render (s) | SR (s) | total ↓ (s) | clean jar ↑ /19 | clean castle ↑ /19 | clean total ↑ /38 | flips vs current (+ / −) |
|---|---|---|---|---|---|---|---|---|
| current 4 nodes | 42.1 | 72.0 | 0.0 | 114.1 | 8 | 6 | <u>14</u> | - |
| 3 nodes | 42.1 | 58.4 | 0.0 | 100.5 | 11 | 4 | **15** | +4 / −3 |
| 2 nodes | 42.1 | 45.7 | 0.0 | 87.8 | 8 | 5 | 13 | +2 / −3 |
| render 576+SR | 42.1 | 28.9 | 0.6 | 71.6 | 8 | 3 | 11 | +4 / −7 |
| render 448+SR | 42.1 | 18.6 | 0.4 | <u>61.1</u> | 4 | 0 | 4 | +1 / −11 |
| plan 448 | 28.9 | 72.0 | 0.0 | 100.9 | 4 | 1 | 5 | +2 / −11 |
| plan+SR only | 42.1 | 0.0 | 1.4 | **43.5** | 7 | 5 | 12 | +4 / −6 |
