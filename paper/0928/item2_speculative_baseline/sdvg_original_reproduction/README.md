# SDVG original code on our two prompts (text-to-video)

Code: archive branch `dev` (the paper's method), unchanged except `sdvg_ours.py` (runner) and a loader fallback in `utils/wan_wrapper.py` that builds the 14B architecture from `config.json` (the Krea checkpoint covers all 1,095 parameters, strict load).
Models: Wan2.1-T2V-1.3B Self-Forcing drafter (`self_forcing_dmd.pt`, 4 steps), Krea Realtime 14B target, ImageReward-v1.0 router (min over frames, fixed τ = −0.7, block 0 force-rejected).
Setting: text-to-video, 13 blocks (150 frames) at 832×480, seeds 1234/4321/9999, one RTX PRO 6000 pair per run, SDPA attention (no FlashAttention/SageAttention wheels for Blackwell), times exclude model loading.
Prompts (no start image, no camera control — this model cannot take either):
- jar: "a glass jar with a funnel on a rustic wooden table, sand pouring into the jar, blurred neutral indoor background, natural light"
- castle: "an elaborate sandcastle on a beach being hit by a large green breaking wave, wet sand and sea foam, overcast sky"

| arm | mean time per clip | speed-up vs target |
|---|---|---|
| target_only_krea14b | 70.4 s | 1.00× |
| sdvg_tau-0.7 | 31.1 s | 2.27× |
| draft_only_selfforcing1p3b | 27.3 s | 2.58× |

| case | arm | time | blocks accepted / rejected |
|---|---|---|---|
| jar s1234 | target_only_krea14b | 77.4 s | 0 / 13 |
| jar s1234 | sdvg_tau-0.7 | 38.5 s | 12 / 1 |
| jar s1234 | draft_only_selfforcing1p3b | 28.4 s | 13 / 0 |
| jar s4321 | target_only_krea14b | 68.2 s | 0 / 13 |
| jar s4321 | sdvg_tau-0.7 | 28.3 s | 12 / 1 |
| jar s4321 | draft_only_selfforcing1p3b | 26.4 s | 13 / 0 |
| jar s9999 | target_only_krea14b | 68.8 s | 0 / 13 |
| jar s9999 | sdvg_tau-0.7 | 28.9 s | 12 / 1 |
| jar s9999 | draft_only_selfforcing1p3b | 26.1 s | 13 / 0 |
| castle s1234 | target_only_krea14b | 69.7 s | 0 / 13 |
| castle s1234 | sdvg_tau-0.7 | 30.1 s | 12 / 1 |
| castle s1234 | draft_only_selfforcing1p3b | 27.5 s | 13 / 0 |
| castle s4321 | target_only_krea14b | 69.9 s | 0 / 13 |
| castle s4321 | sdvg_tau-0.7 | 30.2 s | 12 / 1 |
| castle s4321 | draft_only_selfforcing1p3b | 27.8 s | 13 / 0 |
| castle s9999 | target_only_krea14b | 68.5 s | 0 / 13 |
| castle s9999 | sdvg_tau-0.7 | 30.4 s | 12 / 1 |
| castle s9999 | draft_only_selfforcing1p3b | 27.6 s | 13 / 0 |

Observations (s1234 strips): SDVG shares block 0 with the target run, then follows the drafter; on the castle the target destroys the castle by the end while SDVG keeps it standing (drafter behaviour). Acceptance 12/13 on every clip (ImageReward min-frame 0.07–1.41 on these static-camera T2V clips vs ≈ −2.28 on our camera-pan clips). Paper: 1.59× at 73 % acceptance on A6000.
`side_by_side/<task>_s<seed>_target_sdvg_draft.mp4`: target | SDVG | draft.
