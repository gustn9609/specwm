# SDVG original code (branch dev): drafter-only vs SDVG vs target-only

Text-to-video (no start image, no camera). 13 blocks = 150 frames at 832x480, 16 fps (re-muxed from 24 fps on 9/30; frames unchanged). Seeds 1234/4321/9999. One RTX PRO 6000 pair per run; times exclude model loading.

Prompts:
- jar: "a glass jar with a funnel on a rustic wooden table, sand pouring into the jar, blurred neutral indoor background, natural light"
- castle: "an elaborate sandcastle on a beach being hit by a large green breaking wave, wet sand and sea foam, overcast sky"

| method | mean time per clip | speed-up vs target | blocks accepted (mean) |
|---|---|---|---|
| 1_drafter-only_1.3B | 27.3 s | 2.58x | 13.0 / 13 |
| 2_SDVG_paper_tau-0.7 | 31.1 s | 2.27x | 12.0 / 13 |
| 2b_SDVG_tau-1.3 | 50.7 s | 1.39x | 5.8 / 13 |
| 2c_SDVG_tau-1.5 | 63.2 s | 1.11x | 1.8 / 13 |
| 3_target-only_Krea14B | 70.4 s | 1.00x | 0.0 / 13 |

τ sweep: −0.7 is the paper default (ImageReward min-frame; on these static-camera T2V clips the scores are high, median 1.41, so almost everything passes); 1.3 and 1.5 are stricter thresholds chosen from the logged scores to send roughly 40 % and 70 % of draft blocks to the target.

| case | method | time | blocks accepted / rejected |
|---|---|---|---|
| jar s1234 | 1_drafter-only_1.3B | 28.4 s | 13 / 0 |
| jar s1234 | 2_SDVG_paper_tau-0.7 | 38.5 s | 12 / 1 |
| jar s1234 | 2b_SDVG_tau-1.3 | 67.3 s | 1 / 12 |
| jar s1234 | 2c_SDVG_tau-1.5 | 70.8 s | 0 / 13 |
| jar s1234 | 3_target-only_Krea14B | 77.4 s | 0 / 13 |
| jar s4321 | 1_drafter-only_1.3B | 26.4 s | 13 / 0 |
| jar s4321 | 2_SDVG_paper_tau-0.7 | 28.3 s | 12 / 1 |
| jar s4321 | 2b_SDVG_tau-1.3 | 57.4 s | 4 / 9 |
| jar s4321 | 2c_SDVG_tau-1.5 | 55.2 s | 4 / 9 |
| jar s4321 | 3_target-only_Krea14B | 68.2 s | 0 / 13 |
| jar s9999 | 1_drafter-only_1.3B | 26.1 s | 13 / 0 |
| jar s9999 | 2_SDVG_paper_tau-0.7 | 28.9 s | 12 / 1 |
| jar s9999 | 2b_SDVG_tau-1.3 | 57.4 s | 4 / 9 |
| jar s9999 | 2c_SDVG_tau-1.5 | 68.9 s | 0 / 13 |
| jar s9999 | 3_target-only_Krea14B | 68.8 s | 0 / 13 |
| castle s1234 | 1_drafter-only_1.3B | 27.5 s | 13 / 0 |
| castle s1234 | 2_SDVG_paper_tau-0.7 | 30.1 s | 12 / 1 |
| castle s1234 | 2b_SDVG_tau-1.3 | 30.2 s | 12 / 1 |
| castle s1234 | 2c_SDVG_tau-1.5 | 59.0 s | 3 / 10 |
| castle s1234 | 3_target-only_Krea14B | 69.7 s | 0 / 13 |
| castle s4321 | 1_drafter-only_1.3B | 27.8 s | 13 / 0 |
| castle s4321 | 2_SDVG_paper_tau-0.7 | 30.2 s | 12 / 1 |
| castle s4321 | 2b_SDVG_tau-1.3 | 55.8 s | 4 / 9 |
| castle s4321 | 2c_SDVG_tau-1.5 | 56.1 s | 4 / 9 |
| castle s4321 | 3_target-only_Krea14B | 69.9 s | 0 / 13 |
| castle s9999 | 1_drafter-only_1.3B | 27.6 s | 13 / 0 |
| castle s9999 | 2_SDVG_paper_tau-0.7 | 30.4 s | 12 / 1 |
| castle s9999 | 2b_SDVG_tau-1.3 | 36.3 s | 10 / 3 |
| castle s9999 | 2c_SDVG_tau-1.5 | 69.3 s | 0 / 13 |
| castle s9999 | 3_target-only_Krea14B | 68.5 s | 0 / 13 |

File names: `<scene>__<prompt-slug>__seed<seed>__<method>__<seconds>s.mp4`; `*__side-by-side_5methods.mp4` = drafter | SDVG −0.7 | SDVG 1.3 | SDVG 1.5 | target, labelled.
